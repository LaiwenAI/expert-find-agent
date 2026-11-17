"""Compare exact lexical gates after cached dense candidate generation."""
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from expert_search.cof_benchmark import load
from expert_search.fast_retrieval import FastSearch
from expert_search.resident_candidate_bm25 import ResidentCandidateBM25
from expert_search.shared_owner_gate import SharedOwnerGate


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root,papers,reviewers,labels=load('NIPS');corpus=root/'unique-profile-corpus';db=corpus/'fast.sqlite'
    vectors=root/'qwen-embedding-torch-v1'/'vectors.npy';summary=json.loads((vectors.parent/'summary.json').read_text())
    if sha(vectors)!=summary['vectors_sha256']:raise ValueError('Vector hash mismatch')
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft']);eids=sorted(reviewers)
    matrix=np.load(vectors,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(pids)+len(qids),1024):raise ValueError('Unexpected vector layout')
    positions={pid:i for i,pid in enumerate(pids)}
    owner_rows=np.array([positions[p] for e in eids for p in sorted(set(reviewers[e]['papers']))],dtype=np.int64)
    starts=np.cumsum([0]+[len(set(reviewers[e]['papers'])) for e in eids[:-1]])
    lex=FastSearch(corpus,db);resident=ResidentCandidateBM25(db,adaptive_intersection=True);gate=SharedOwnerGate(resident,{e:r['papers'] for e,r in reviewers.items()})
    def candidate(qi):
        values=np.einsum('ij,j->i',matrix[:len(pids)],matrix[len(pids)+qi],optimize=False)
        best=np.maximum.reduceat(values[owner_rows],starts)
        return eids[int(np.argmax(best))]
    observations=[]
    for qi,qid in enumerate(qids[:30]):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        results=[]
        for repeat in range(3):
            order=('full','bounded') if (qi+repeat)%2==0 else ('bounded','full');pair={}
            for arm in order:
                start=time.perf_counter();eid=candidate(qi)
                if arm=='bounded':
                    result=gate.contains(query,eid,50);accepted=result['accepted'];possible=result['possible_competitors']
                else:
                    scores=lex.document_scores(query)
                    per={e:max((scores.get(pid,0.) for pid in set(r['papers'])),default=0.) for e,r in reviewers.items()}
                    ranked=sorted(per,key=lambda e:(-per[e],e));accepted=eid in ranked[:50] and per[eid]>0;possible=None
                pair[arm]={'seconds':time.perf_counter()-start,'candidate':eid,'accepted':accepted,'possible':possible}
            if pair['full']['candidate']!=pair['bounded']['candidate'] or pair['full']['accepted']!=pair['bounded']['accepted']:
                raise ValueError('Different candidates or gate decisions')
            results.append(pair)
        observations.append({'full_seconds':statistics.mean(v['full']['seconds'] for v in results),
                             'bounded_seconds':statistics.mean(v['bounded']['seconds'] for v in results),
                             'accepted':results[0]['full']['accepted'],'possible':results[0]['bounded']['possible']})
    report={'queries':len(observations),'reviewers':len(reviewers),'sources':len(pids),'paired_attempts':3*len(observations),
            'matching_decisions':3*len(observations),'accepted':sum(r['accepted'] for r in observations),
            'mean_full_seconds':statistics.mean(r['full_seconds'] for r in observations),
            'mean_bounded_seconds':statistics.mean(r['bounded_seconds'] for r in observations),
            'mean_possible_competitors':statistics.mean(r['possible'] for r in observations),
            'resident_init_seconds':resident.initialization_seconds,'bound_init_seconds':gate.initialization_seconds,
            'inputs_sha256':{'vectors':sha(vectors),'index':sha(db),'papers':sha(root/'NIPS_papers_test.json'),
              'reviewers':sha(root/'NIPS_reviewers_test.json'),'labels':sha(root/'NIPS_queries_test_soft.json'),
              'gate':sha(Path('src/expert_search/shared_owner_gate.py')),'script':sha(Path(__file__))},
            'scope':'Exploratory first 30 previously used public CoF NIPS queries. Each timed arm recomputes identical cached-vector dense top-one and its lexical membership decision; score-only full path and bounded path have different outputs. Cached query vectors exclude online encoding. Three alternating observations per query are dependent; no independent warmup, contention or service latency. Index/vector/model initialization, source creation, inference reasoning and verification excluded. Matching membership is not evidence entailment, reviewer relevance or enterprise fit.'}
    out=Path('artifacts/cof-dense-gate-pipeline-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
