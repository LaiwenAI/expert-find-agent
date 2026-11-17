"""Check selective shortlist rankings against full document-score replay."""
import hashlib
import json
import math
from pathlib import Path
from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch


def main():
    out=Path('artifacts/cof-sigir-selective-rerank-v1');summary=json.loads((out/'summary.json').read_text());rows=json.loads((out/'results.json').read_text())
    root,papers,reviewers,labels=load('SIGIR')
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=value for name,value in summary['input_sha256'].items()):raise ValueError('Changed source')
    ids=sorted({pid for r in reviewers.values() for pid in r['papers']});documents={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in ids}
    profiles={eid:sorted(set(r['papers'])) for eid,r in reviewers.items()};model=CompiledCandidateLM(ExpertLanguageModels(documents,profiles));search=FastSearch(root/'unique-profile-corpus',out/'selective.sqlite')
    for _,body in search.db.execute('SELECT id,body FROM docs'):
        row=json.loads(body)
        if row['id'] not in documents or row['text']!=documents[row['id']] or row['expert_id']!=row['id']:raise ValueError('Indexed source differs')
    if len(search.metadata)!=len(documents):raise ValueError('Incomplete indexed sources')
    if len(rows)!=73 or {r['query_id'] for r in rows}!=set(labels['soft']):raise ValueError('Changed query set')
    checks=0;max_error=0.
    for row in rows:
        qid=row['query_id'];query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip();global_scores=search.document_scores(query)
        for n in (20,50):
            pool=[r['id'] for r in certified_topk(model,query,n)['results']]
            source_ids=sorted({pid for eid in pool for pid in profiles[eid]});selected=search.evidence_for_selective(query,source_ids,1)
            for pid in source_ids:
                if len(selected[pid])!=1:raise ValueError('Unexpected source evidence')
                max_error=max(max_error,abs(selected[pid][0]['score']-global_scores.get(pid,0.)))
                if not math.isclose(selected[pid][0]['score'],global_scores.get(pid,0.),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Selective score differs')
            values={eid:max((global_scores.get(pid,0.) for pid in profiles[eid]),default=0.) for eid in pool}
            expected=sorted(pool,key=lambda eid:(-values[eid],eid))[:10]
            if expected!=row['rankings'][f'candidate{n}_bm25']:raise ValueError('Shortlist ranking differs')
            checks+=1
    search.db.close();print(json.dumps(dict(queries=len(rows),ranking_checks=checks,max_absolute_source_score_difference=max_error,
                                           scope='Verifies scored source texts and shortlist ranking against same compiled global scores; not relevance correctness, source authenticity or latency stability.'),indent=2))


if __name__=='__main__':main()
