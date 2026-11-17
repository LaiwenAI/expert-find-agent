"""Measure model-free lexical span-proposal coverage on development needs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.lexical_anchored_choice_agent import propose_pair


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');manifest=json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl')!=manifest['files_sha256']['queries.jsonl']:raise ValueError('Changed queries')
    queries=[r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development']
    if len(queries)!=80 or len({q['id'] for q in queries})!=80:raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'queries':80,'queries_sha256':sha(data/'queries.jsonl'),'candidate_pool':50,'candidates':5,'source_chars':1200,'span_width':400,
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/lexical_anchored_choice_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Private development needs; lexical overlap is a proposal, not evidence of semantic match, relevance, personal contribution, or inability. No model calls or protected split.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start
    rows=[];coverage=Counter()
    for i,q in enumerate(queries):
        start=time.perf_counter();retrieved=engine.search_reranked(q['text'],k=5,candidates=50,evidence_per_expert=1);retrieval=time.perf_counter()-start
        start=time.perf_counter();pairs=[]
        for row in retrieved['results']:
            expert=row['expert']['id']
            if any(s['expert_id']!=expert for s in row['evidence']):raise ValueError('Foreign evidence')
            pair=propose_pair(q['text'],row['evidence'],width=400,source_chars=1200);pairs.append(pair)
            if pair:
                d=pair['demand_anchor'];s=pair['source_anchor'];source=next(e for e in row['evidence'] if e['id']==s['evidence_id'])
                if q['text'][d['start']:d['end']]!=d['quote'] or source['text'][s['start']:s['end']]!=s['quote'] or s['expert_id']!=expert:raise ValueError('Invalid proposed span')
        proposal=time.perf_counter()-start;n=sum(p is not None for p in pairs);coverage['candidates']+=len(pairs);coverage['with_overlap']+=n;coverage['without_overlap']+=len(pairs)-n;coverage['queries_all_covered']+=n==len(pairs);coverage['queries_no_overlap']+=n==0
        rows.append({'query_id':q['id'],'candidate_count':len(pairs),'proposed':n,'retrieval_seconds':retrieval,'proposal_seconds':proposal})
        if (i+1)%20==0:print('Completed',i+1,flush=True)
    engine.evidence.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'queries':len(rows),'counts':dict(coverage),'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),
            'mean_proposal_seconds':statistics.mean(r['proposal_seconds'] for r in rows),'pipeline_init_seconds':init,
            'corpus_snapshot_sha256':engine.ranker.snapshot,'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'private-results.json'),
            'scope':protocol['scope']+' One sequential timing per query excludes initialization and model inference. Candidate/view IDs and private source text are not published.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
