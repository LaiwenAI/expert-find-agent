"""Development-only lexical discovery outside a frozen dense-selected pool.

Does not alter snapshots, labels or heldout data; discovered candidates stay unjudged.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from expert_search.retrieval import tokens


def run():
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    out=Path('artifacts/fresh-lexical-pool-audit-v1');out.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    names=['queries.jsonl','pools.jsonl','corpus.jsonl']
    for name in names:
        with (root/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=manifest['files_sha256'][name]:raise ValueError('Changed frozen input')
    queries=[q for q in map(json.loads,(root/'queries.jsonl').read_text().splitlines()) if q['split']=='development']
    if len(queries)!=80:raise ValueError('Wrong development subset')
    pools={p['query_id']:p for p in map(json.loads,(root/'pools.jsonl').read_text().splitlines()) if p['split']=='development'}
    query_terms=set(t for q in queries for t in tokens(q['text']))
    postings=defaultdict(list);docs=[];lengths=[];start=time.perf_counter()
    for row in map(json.loads,(root/'corpus.jsonl').read_text().splitlines()):
        if row['split']!='development':continue
        counts=Counter(tokens(row['text'][:1200]));i=len(docs)
        docs.append({k:row[k] for k in ('id','expert_id','kind')});lengths.append(sum(counts.values()))
        for t in counts.keys()&query_terms:postings[t].append((i,counts[t]))
    n=len(docs);avg=sum(lengths)/n;prepare=time.perf_counter()-start
    rows=[]
    for q in queries:
        for variant,text in [('full-demand',q['text']),('demand-title',q['text'].splitlines()[0])]:
            begin=time.perf_counter();scores=defaultdict(float)
            for t in set(tokens(text)):
                hits=postings.get(t,[]);idf=math.log(1+(n-len(hits)+.5)/(len(hits)+.5))
                for i,tf in hits:scores[i]+=idf*tf*2.2/(tf+1.2*(.25+.75*lengths[i]/avg))
            owners=defaultdict(list)
            for i,score in scores.items():owners[docs[i]['expert_id']].append((score,docs[i]['id']))
            ranked=[]
            for eid,hits in owners.items():
                hits.sort(key=lambda x:(-x[0],x[1]));ranked.append(dict(expert_id=eid,score=hits[0][0],source_ids=[s for _,s in hits[:3]]))
            ranked.sort(key=lambda x:(-x['score'],x['expert_id']));ranked=ranked[:5]
            pool=set(pools[q['id']]['candidates']);outside=[r['expert_id'] for r in ranked if r['expert_id'] not in pool]
            rows.append(dict(query_id=q['id'],variant=variant,ranking=ranked,outside_frozen_pool=outside,seconds=time.perf_counter()-begin,label_status='unjudged_discovery_not_relevance_gold'))
    body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(out/'runs.jsonl').write_text(body)
    summary=dict(development_queries=len(queries),heldout_queries_used=0,development_sources=n,index_preparation_seconds=prepare,
                 inputs_sha256={name:manifest['files_sha256'][name] for name in names},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 runs_sha256=hashlib.sha256(body.encode()).hexdigest(),variants={})
    for variant in ['full-demand','demand-title']:
        selected=[r for r in rows if r['variant']==variant]
        summary['variants'][variant]=dict(returned=sum(len(r['ranking']) for r in selected),outside_pool_positions=sum(len(r['outside_frozen_pool']) for r in selected),queries_with_outside_candidates=sum(bool(r['outside_frozen_pool']) for r in selected),mean_seconds=sum(r['seconds'] for r in selected)/len(selected))
    summary['scope']='Lexical candidate diversity audit, no API calls or new relevance labels. BM25-max, first1200 source characters, development-only DF/length statistics, unique query terms. Titles are original first lines, not generated technical decompositions. Timing local exploratory CPU, not paired end-to-end speed comparison.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
