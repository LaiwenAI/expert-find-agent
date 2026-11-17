"""Replay fixed source vectors for two explicit within-pool dense baselines."""
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    root=Path('artifacts/fresh-method-data-v1');jobs=[json.loads(s) for s in (root/'evaluation-jobs.jsonl').read_text().splitlines()]
    pool=json.loads((root/'pool-protocol.json').read_text());source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=pool['source_sha256']:raise ValueError('Changed source order')
    with vp.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=pool['vectors_sha256']:raise ValueError('Changed source vectors')
    positions={json.loads(s)['id']:i for i,s in enumerate(source.read_text().splitlines())};matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    with np.load(root/'query-vectors.npz',allow_pickle=False) as f:qvectors=f['vectors'];qids=f['ids']
    if list(qids)!=[j['id'] for j in jobs]:raise ValueError('Changed query ordering')
    runs=[]
    for job,q in zip(jobs,qvectors):
        scores={m:{} for m in ['qwen-paper-max','qwen-view-max']}
        for eid,sources in job['candidates'].items():
            selected=[positions[sid] for sid in sources];values=np.einsum('ij,j->i',matrix[selected],q,optimize=False);by_id=dict(zip(sources,map(float,values)))
            scores['qwen-view-max'][eid]=max(by_id.values());scores['qwen-paper-max'][eid]=max(v for sid,v in by_id.items() if sources[sid]['kind']=='paper')
        for method,values in scores.items():runs.append(dict(query_id=job['id'],split=job['split'],method=method,ranking=sorted(values,key=lambda e:(-values[e],e)),scores=values))
    body=''.join(json.dumps(r)+'\n' for r in runs);path=root/'dense-baseline-runs.jsonl'
    if path.exists() and path.read_text()!=body:raise ValueError('Changed frozen baseline runs')
    path.write_text(body);(root/'dense-baseline-protocol.json').write_text(json.dumps(dict(jobs_sha256=hashlib.sha256((root/'evaluation-jobs.jsonl').read_bytes()).hexdigest(),vectors_sha256=pool['vectors_sha256'],query_vectors_sha256=hashlib.sha256((root/'query-vectors.npz').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Same7candidate source snapshots. Paper-max versus all-visible-source-max with cached query/source vectors. Full raw source text beyond snapshot prefix not used; offline replay, not online latency. No label values read.'),indent=2));print('Stored',len(runs),'fixed dense baseline rankings')


if __name__=='__main__':run()
