"""Compare bounded membership with full scoring on a larger public corpus."""
import hashlib
import json
import statistics
import time
from pathlib import Path
from expert_search.cof_benchmark import load
from expert_search.fast_retrieval import FastSearch
from expert_search.resident_candidate_bm25 import ResidentCandidateBM25
from expert_search.shared_owner_gate import SharedOwnerGate


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root,papers,reviewers,labels=load('NIPS')
    corpus=root/'unique-profile-corpus';db=corpus/'fast.sqlite'
    index_hash=sha(db);lexical=FastSearch(corpus,db)
    resident=ResidentCandidateBM25(db,adaptive_intersection=True)
    gate=SharedOwnerGate(resident,{eid:r['papers'] for eid,r in reviewers.items()})
    qids=sorted(labels['soft'])[:30]
    if len(qids)!=30 or len(reviewers)<100:raise ValueError('Insufficient corpus')
    rows=[]
    for i,qid in enumerate(qids):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        scores=lexical.document_scores(query)
        owner_scores={eid:max((scores.get(pid,0.) for pid in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
        ranking=sorted(owner_scores,key=lambda e:(-owner_scores[e],e))
        for position in (9,49,99):
            candidate=ranking[position];observations=[]
            for repeat in range(3):
                order=('full','bounded') if (i+repeat+position)%2==0 else ('bounded','full')
                result={}
                for arm in order:
                    start=time.perf_counter()
                    if arm=='bounded':
                        value=gate.contains(query,candidate,50);decision=value['accepted'];possible=value['possible_competitors']
                    else:
                        values=lexical.document_scores(query)
                        scores2={eid:max((values.get(pid,0.) for pid in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
                        order2=sorted(scores2,key=lambda eid:(-scores2[eid],eid))
                        decision=candidate in order2[:50] and scores2[candidate]>0;possible=None
                    result[arm]={'seconds':time.perf_counter()-start,'accepted':decision,'possible':possible}
                if result['full']['accepted']!=result['bounded']['accepted']:raise ValueError('Gate disagreement')
                observations.append(result)
            rows.append({'position':position+1,'full_seconds':statistics.mean(o['full']['seconds'] for o in observations),
                         'bounded_seconds':statistics.mean(o['bounded']['seconds'] for o in observations),
                         'possible_competitors':observations[0]['bounded']['possible'],'accepted':observations[0]['bounded']['accepted']})
    if sha(db)!=index_hash:raise ValueError('Index changed')
    report={'queries':len(qids),'reviewers':len(reviewers),'sources':len(resident.ids),'cases':len(rows),'paired_attempts':len(rows)*3,
            'resident_initialization_seconds':resident.initialization_seconds,'bound_initialization_seconds':gate.initialization_seconds,
            'cases_by_original_rank':{str(pos):{'cases':len(group),'accepted':sum(r['accepted'] for r in group),
                  'mean_full_seconds':statistics.mean(r['full_seconds'] for r in group),'mean_bounded_seconds':statistics.mean(r['bounded_seconds'] for r in group),
                  'mean_possible_competitors':statistics.mean(r['possible_competitors'] for r in group)} for pos in (10,50,100) for group in [[r for r in rows if r['position']==pos]]},
            'input_sha256':{'index':index_hash,'papers':sha(root/'NIPS_papers_test.json'),'reviewers':sha(root/'NIPS_reviewers_test.json'),
                'labels':sha(root/'NIPS_queries_test_soft.json'),'gate':sha(Path('src/expert_search/shared_owner_gate.py')),'script':sha(Path(__file__))},
            'scope':'Exploratory previously available CoF NIPS reviewer queries, first 30 sorted IDs; candidates chosen using full ranking, outside reported timing. Three alternating sequential component timings per candidate are dependent; no separate warmup, traffic or concurrency. Index initialization and original candidate generation excluded from query time. This checks only exact numerical membership, not evidence support, source ownership truth, relevance, industrial expert fit, or production latency. Timing includes full source ranking versus bounded membership, not the same output contract.'}
    out=Path('artifacts/cof-shared-gate-scale-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
