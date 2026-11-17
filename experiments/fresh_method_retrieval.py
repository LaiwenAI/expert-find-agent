"""Exact paper-backed retrieval for the newly frozen, prior-disjoint demand cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.qwen_embedding_torch import QwenTorchEmbedding

ROOT=Path('artifacts/fresh-method-data-v1')
TASK='Find researchers whose documented research background could help address the technical requirement.'


def nh(text):
    import re
    return hashlib.sha256(re.sub(r'[^\w]+','',text.casefold()).encode()).hexdigest()


def run(stage):
    queries=[json.loads(s) for s in (ROOT/'queries.jsonl').read_text().splitlines()]
    old=json.loads((ROOT/'prior-use.json').read_text());parent=Path('artifacts/global-expert-branch-benchmark-v1')
    base=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    protocol=dict(queries_sha256=hashlib.sha256((ROOT/'queries.jsonl').read_bytes()).hexdigest(),registry_sha256=hashlib.sha256((ROOT/'prior-use.json').read_bytes()).hexdigest(),
                  source_sha256=base['roots_sha256'],vectors_sha256=summary['vectors_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  task=TASK,query_max_length=4096,source_max_tokens=512,source_prefix_chars=1200,
                  rule='All100 fixed unseen demands, exact paper-only max-source expert selection; exclude registered prior experts, exact source prefix hashes, normalized source text and paper-title aliases. Top1 remaining expert/top3 eligible same-owner papers, no outcome filter.',
                  scope='New candidate/source collection, not expert-fit labels. Paper-backed tasks; profile-only experts not treated as incapable. Prior-disjoint exact groups do not establish semantic/team independence. No model training.')
    marker=ROOT/'retrieval-protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed fresh retrieval protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    if stage=='encode':
        model_root=Path('artifacts/models/qwen3-embedding-0.6b')
        for name,expected in base['checkpoint']['files'].items():
            with (model_root/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Changed official encoder')
        model=QwenTorchEmbedding(max_length=4096,batch_size=8)
        start=time.perf_counter();vectors=model.encode(['Instruct: '+TASK+'\nQuery:'+q['demand'] for q in queries]);seconds=time.perf_counter()-start
        np.savez(ROOT/'query-vectors.npz',vectors=vectors,ids=np.asarray([q['id'] for q in queries]),seconds=seconds)
        print('Encoded new demands:',len(queries),flush=True);return
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=parent/'vectors.npy'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=base['roots_sha256']:raise ValueError('Changed source snapshot')
    with vp.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=summary['vectors_sha256']:raise ValueError('Changed vectors')
    records=[json.loads(s) for s in source.read_text().splitlines()];full=np.load(vp,mmap_mode='r',allow_pickle=False)
    used_experts=set(old['expert_ids']);used_prefixes=set(old['source_prefix_hashes']);used_normalized=set(old['normalized_source_hashes']);used_titles=set(old['paper_title_hashes'])
    positions=[]
    for i,r in enumerate(records):
        if r['kind']!='paper' or r['expert_id'] in used_experts:continue
        text=r['text'];title=text.splitlines()[0] if text else ''
        if hashlib.sha256(text.encode()).hexdigest() in used_prefixes or nh(text) in used_normalized or nh(title) in used_titles:continue
        positions.append(i)
    if not positions:raise ValueError('No new eligible paper sources')
    papers=[records[i] for i in positions];matrix=np.ascontiguousarray(full[positions]);by_owner={}
    for i,paper in enumerate(papers):by_owner.setdefault(paper['expert_id'],[]).append(i)
    with np.load(ROOT/'query-vectors.npz',allow_pickle=False) as f:vectors=f['vectors'];ids=f['ids']
    if list(ids)!=[q['id'] for q in queries] or vectors.shape!=(len(queries),1024):raise ValueError('New query inventory mismatch')
    jobs=[]
    for query,q in zip(queries,vectors):
        scores=np.einsum('ij,j->i',matrix,q,optimize=False)
        best=min(range(len(papers)),key=lambda i:(-float(scores[i]),papers[i]['expert_id'],papers[i]['id']));eid=papers[best]['expert_id']
        chosen=sorted(by_owner[eid],key=lambda i:(-float(scores[i]),papers[i]['id']))[:3]
        jobs.append(dict(id=query['id']+'-expert-'+eid,parent_demand=query['id'],demand=query['demand'],demand_url=query['url'],demand_source_row=query['source_row'],expert_id=eid,sources={papers[i]['id']:papers[i] for i in chosen},retrieval_scores={papers[i]['id']:float(scores[i]) for i in chosen},status='new_retrieved_unlabeled'))
    body=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs);path=ROOT/'retrieved-jobs.jsonl'
    if path.exists() and path.read_text()!=body:raise ValueError('Changed fresh retrieval jobs')
    path.write_text(body)
    report=dict(new_demands=len(jobs),new_candidate_experts=len({j['expert_id'] for j in jobs}),eligible_paper_sources=len(papers),eligible_experts=len(by_owner),selected_source_ids=len({sid for j in jobs for sid in j['sources']}),jobs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                warning='All100 new source jobs retained; no ability, relevance or future-project labels. Exact exclusion registry, semantic near duplicates still unverified.')
    (ROOT/'retrieval-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['encode','retrieve'],required=True);run(p.parse_args().stage)
