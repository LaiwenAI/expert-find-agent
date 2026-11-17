"""Verify frozen author-recovery data without treating absent authors as negatives."""
import argparse
import hashlib
import json
from pathlib import Path

from .build_author_recovery import NON_ENGLISH_MARKS,key,similarity,title_grams


def verify(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());files=manifest['outputs_sha256']
    expected={'experts.jsonl','evidence.jsonl','queries.jsonl','qrels.jsonl','source-records.jsonl'}
    if set(files)!=expected:raise ValueError('Unexpected dataset file manifest')
    for name,digest in files.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed data: '+name)
    def read(name):return [json.loads(line) for line in (root/name).read_text().splitlines()]
    source=read('source-records.jsonl');by_id={r['id']:r for r in source}
    if len(by_id)!=len(source):raise ValueError('Duplicate source work')
    for r in source:
        if r.get('language')!='en' or NON_ENGLISH_MARKS.search(r['title']) or not key(r['title']) or r['authors']!=sorted(set(r['authors'])):raise ValueError('Invalid source metadata')
        if r['period']=='history' and not 2010<=r['year']<=2019:raise ValueError('Invalid history year')
        if r['period']=='target' and not 2020<=r['year']<=2023:raise ValueError('Invalid target year')
        if r['period'] not in ('history','target'):raise ValueError('Invalid period')
    history=[r for r in source if r['period']=='history'];targets=[r for r in source if r['period']=='target']
    if {key(r['title']) for r in history}&{key(r['title']) for r in targets}:raise ValueError('Exact target title leakage')
    target_grams=[title_grams(r['title']) for r in targets]
    history_grams=[title_grams(r['title']) for r in history]
    if any(similarity(h,g)>=.8 for h in history_grams for g in target_grams):raise ValueError('High-overlap target title leakage')
    if len({key(r['title']) for r in targets})!=len(targets):raise ValueError('Duplicate target titles')
    experts=read('experts.jsonl');ids=[r['id'] for r in experts];eligible={a for r in history for a in r['authors']}
    if len(ids)!=len(set(ids)) or set(ids)!=eligible:raise ValueError('Invalid expert inventory')
    evidence=read('evidence.jsonl');expected_pairs={(r['id'],a) for r in history for a in r['authors']}
    actual_pairs={(r['source_work_id'],r['expert_id']) for r in evidence}
    if len(evidence)!=len(actual_pairs) or actual_pairs!=expected_pairs:raise ValueError('Invalid evidence ownership')
    for r in evidence:
        work=by_id[r['source_work_id']]
        if r['id']!=work['id']+':'+r['expert_id'] or r['text']!=work['title'] or r['publication_year']!=work['year']:raise ValueError('Changed evidence')
    queries=read('queries.jsonl');qrels=read('qrels.jsonl');qs={r['id']:r for r in queries};labels={r['query_id']:r['known_author_ids'] for r in qrels}
    expected_queries={r['id'] for r in targets if set(r['authors'])&eligible}
    if len(qs)!=len(queries) or len(labels)!=len(qrels) or set(qs)!=expected_queries or set(labels)!=expected_queries:raise ValueError('Invalid query coverage')
    for qid,row in qs.items():
        target=by_id[qid];expected_labels=sorted(set(target['authors'])&eligible)
        if row['text']!=target['title'] or row['publication_year']!=target['year'] or labels[qid]!=expected_labels:raise ValueError('Invalid known-author association')
    if (manifest['queries'],manifest['experts'],manifest['evidence'],manifest['history_works'],manifest['target_works'])!=(len(qs),len(ids),len(evidence),len(history),len(targets)):raise ValueError('Manifest count mismatch')
    return {'queries':len(qs),'experts':len(ids),'evidence':len(evidence),'history_works':len(history),'target_works':len(targets),
            'scope':'Verifies frozen derivation, time separation and a character 4-gram overlap threshold for released titles; not source authenticity, semantic leakage, author disambiguation, individual contribution or enterprise relevance.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path,nargs='?',default=Path('benchmarks/openalex-author-recovery-v1'));args=parser.parse_args()
    print(json.dumps(verify(args.root),indent=2))
