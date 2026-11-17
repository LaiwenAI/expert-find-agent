"""Analyze query-vector cosine against bilingual ranking overlap on all intents."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.bilingual.analyze_alignment_dependence import analyze


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    root=Path('artifacts/bilingual-intent-retrieval-results-v1');summary=json.loads((root/'summary.json').read_text())
    for name,key in (('protocol.json','protocol_sha256'),('private-results.json','private_results_sha256'),('query-vectors.npy','query_vectors_sha256')):
        if sha(root/name)!=summary[key]:raise ValueError('Changed input: '+name)
    protocol=json.loads((root/'protocol.json').read_text());selection=Path('artifacts/bilingual-intent-retrieval-v1');sel=json.loads((selection/'selection.json').read_text())
    if sha(selection/'selection.json')!=protocol['selection_sha256'] or sha(selection/'queries.jsonl')!=sel['query_sha256'] or protocol['query_sha256']!=sel['query_sha256']:raise ValueError('Changed frozen query selection')
    queries=[json.loads(x) for x in (selection/'queries.jsonl').read_text().splitlines()];results=json.loads((root/'private-results.json').read_text())
    vectors=np.load(root/'query-vectors.npy',mmap_mode='r',allow_pickle=False)
    if len(queries)!=5000 or len(results)!=5000 or vectors.shape!=(10000,1024) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-5):raise ValueError('Invalid cached queries or vectors')
    rows=[]
    for i,(q,r) in enumerate(zip(queries,results)):
        if any(q[k]!=r[k] for k in ('id','parent_id','group_id','intent_type','status')):raise ValueError('Changed query vector ordering')
        x,y=r['ranks']['zh'],r['ranks']['en']
        if any(len(a)!=10 or len(set(a))!=10 for a in (x,y)):raise ValueError('Invalid rankings')
        rows.append(dict(id=r['id'],group_id=r['group_id'],intent_type=r['intent_type'],status=r['status'],query_cosine=float(vectors[2*i]@vectors[2*i+1]),language_overlap={'exact':len(set(x)&set(y))/10}))
    report=analyze(rows)
    strata={}
    for kind in sorted({r['intent_type'] for r in rows}):
        subset=[r for r in rows if r['intent_type']==kind];cos=np.asarray([r['query_cosine'] for r in subset]);overlap=np.asarray([r['language_overlap']['exact'] for r in subset])
        strata[kind]=dict(queries=len(subset),mean_cosine=float(cos.mean()),mean_top10_overlap=float(overlap.mean()),pearson_r=float(np.corrcoef(cos,overlap)[0,1]))
    report.update(by_intent=strata,input_sha256={str(p):sha(p) for p in (root/'summary.json',root/'protocol.json',root/'private-results.json',root/'query-vectors.npy',selection/'selection.json',selection/'queries.jsonl')},script_sha256=sha(__file__),base_analyzer_sha256=sha('experiments/bilingual/analyze_alignment_dependence.py'),
        scope='Post-hoc association on all construction pairs with company-group bootstrap. Five dependent intents per parent and repeated company groups are resampled together. Descriptive type results are unadjusted; query cosine requires both language views. Neither cosine nor ranking overlap establishes semantic equivalence, industrial expert relevance, causation, or held-out performance. No source text or identifiers in aggregate.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
