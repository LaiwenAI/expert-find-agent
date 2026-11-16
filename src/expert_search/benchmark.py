"""Deterministic held-out-paper author recovery benchmark (proxy, not gold relevance)."""
import hashlib
import json
import math
import random
import time
from pathlib import Path
from .corpus import records
from .retrieval import Search


def metrics(ranking, qrels, k=10):
    ids=ranking[:k]
    rel=[qrels.get(x,0) for x in ids]
    dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(rel))
    ideal=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(qrels.values(),reverse=True)[:k]))
    positives={x for x,g in qrels.items() if g>0}
    return {'ndcg@10':dcg/ideal if ideal else 0,
            'mrr@10':next((1/(i+1) for i,g in enumerate(rel) if g>0),0),
            'recall@10':len(set(ids)&positives)/len(positives) if positives else 0}


def run(corpus='artifacts/corpus', database='artifacts/search.sqlite', output='artifacts/benchmark', n=30, seed=42):
    all_rows=[r for _,r in records(Path(corpus)/'evidence.jsonl')]
    papers=[r for r in all_rows if r['kind']=='paper' and len(r.get('title') or '')>=30]
    random.Random(seed).shuffle(papers)
    selected=[]; seen=set()
    for p in papers:
        if p['expert_id'] not in seen:
            selected.append(p); seen.add(p['expert_id'])
        if len(selected)==n: break
    if len(selected)<n: raise ValueError('Insufficient unique authors')
    # Exclude every exact duplicate title for every query; affiliations and author names never enter queries.
    held_titles={p['title'].strip().casefold() for p in selected}
    excluded={r['id'] for r in all_rows if any(title in r['text'].casefold() for title in held_titles)}
    engine=Search(corpus,database)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,p in enumerate(selected):
        title=p['title'].strip().casefold()
        qrels={x['expert_id']:1 for x in papers if x['title'].strip().casefold()==title}
        for method in ['bm25-max','bm25-vote','tfidf-max']:
            start=time.perf_counter()
            result=engine.search(p['title'],10,method,excluded)
            rows.append(dict(query_id=f'paper-{i}',query=p['title'],qrels=qrels,method=method,
                seconds=time.perf_counter()-start,ranking=[x['id'] for x in result],
                metrics=metrics([x['id'] for x in result],qrels)))
    summary={m:{**{key:sum(r['metrics'][key] for r in rows if r['method']==m)/n for key in rows[0]['metrics']},
                'mean_seconds':sum(r['seconds'] for r in rows if r['method']==m)/n}
             for m in ['bm25-max','bm25-vote','tfidf-max']}
    manifest=dict(task='held-out-title author recovery proxy',seed=seed,queries=n,
        excluded_evidence=sorted(excluded),corpus_sha256=hashlib.sha256((Path(corpus)/'evidence.jsonl').read_bytes()).hexdigest(),
        limitations=['Known authors are incomplete relevance labels.','Exact title copies excluded across all evidence fields; paraphrases and related work remain.',
                    'BM25 collection statistics include excluded documents; excluded documents cannot contribute scores.',
                    'This does not evaluate industrial-demand expert relevance.'],summary=summary)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'runs.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    return manifest
