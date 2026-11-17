"""Compare live assessment labels and evidence spans across execution modes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.expert_catalog import ExpertCatalog
from expert_search.grounded_pipeline import GroundedPipeline
from expert_search.span_applicability import SpanApplicability
from expert_search.statistics import paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class FrozenIndex:
    def __init__(self):self.hits=[]
    def search_vector(self,query,k):return {'hits':self.hits[:k]}


class CachedEncoder:
    def encode(self,texts,query=False):return np.array([[1.]],dtype=np.float32)


class SingleRequirementPlanner:
    def plan(self,demand):return {'requirements':{'r':{'text':demand}},'notice':'Full synthetic request is one provisional requirement, not an atomic or complete decomposition.'}


def main(output):
    root=Path('artifacts/bilingual-retrieval-v1');summary=json.loads((root/'summary.json').read_text());selection=json.loads((root/'selection.json').read_text())
    if sha(root/'evaluation.jsonl')!=summary['evaluation_sha256'] or sha(root/'protocol.json')!=summary['protocol_sha256'] or sha(root/'queries.jsonl')!=selection['queries_sha256']:raise ValueError('Changed frozen inputs')
    records=[json.loads(x) for x in (root/'evaluation.jsonl').read_text().splitlines()];queries=[json.loads(x) for x in (root/'queries.jsonl').read_text().splitlines()]
    if len(records)!=1000 or [r['id'] for r in records]!=[r['id'] for r in queries]:raise ValueError('Changed query inventory')
    chosen=sorted(random.Random(20261126).sample(range(len(records)),6));output.mkdir(parents=True,exist_ok=False)
    files=(Path(__file__),Path('src/expert_search/grounded_pipeline.py'),Path('src/expert_search/span_applicability.py'),Path('src/expert_search/source_applicability.py'),Path('src/expert_search/expert_catalog.py'))
    protocol=dict(seed=20261126,queries=6,candidates=3,k=3,workers=[1,3],model='gpt-6-luna',max_retries=0,
        evaluation_sha256=sha(root/'evaluation.jsonl'),queries_sha256=sha(root/'queries.jsonl'),code_sha256={str(p):sha(p) for p in files},
        selection='Six fixed-seed previously explored synthetic Chinese construction intents; fixed top-three exact-vector candidates and first source each.',
        order='Alternating sequential/concurrent by query; one request per candidate per arm; no retries.',
        scope='Independent repeat calls with frozen candidates and single-requirement mock planner; level and exact source-span agreement are structural consistency, not semantic accuracy. Order, timing, model stochasticity and concurrency are confounded; no causal claim.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter();catalog=ExpertCatalog();catalog_init=time.perf_counter()-start
    index=FrozenIndex();encoder=CachedEncoder();planner=SingleRequirementPlanner()
    def make_assessor():return SpanApplicability(LaiwenAI(model='gpt-6-luna',max_retries=0))
    rows=[]
    for n,i in enumerate(chosen):
        q=queries[i];r=records[i]['languages']['zh']['exact'];ids=r['ranking'][:3];sources=r['top_source_ids'][:3]
        index.hits=[dict(id=e,evidence=[dict(id=s)]) for e,s in zip(ids,sources)]
        arms={}
        for arm in (('sequential','concurrent') if n%2==0 else ('concurrent','sequential')):
            pipeline=GroundedPipeline(encoder,index,catalog,planner,make_assessor(),assessor_factory=make_assessor)
            start=time.perf_counter();events=pipeline.run(q['zh'],k=3,candidates=3,assessment_workers=1 if arm=='sequential' else 3);elapsed=time.perf_counter()-start
            assessments=[x for x in events if x['stage']=='assessment'];final=events[-1]
            if len(assessments)!=3 or final['stage']!='result':raise ValueError('Incomplete attempted cohort')
            decisions={}
            for item in assessments:
                outcome=item['outcome'];entry={'status':outcome['status']}
                if outcome['status']=='structurally_valid':
                    assessment=outcome['result']['assessments']['r']
                    entry.update(level=assessment['level'],citations=sorted((c['evidence_id'],c['start'],c['end']) for c in assessment['citations']),assumption_count=len(assessment['assumptions']))
                else:entry['error_type']=outcome.get('error_type')
                decisions[item['expert_id']]=entry
            arms[arm]=dict(seconds=elapsed,status=final['status'],valid=sum(x['outcome']['status']=='structurally_valid' for x in assessments),
                errors=[x['outcome'].get('error_type') for x in assessments if x['outcome']['status']!='structurally_valid'],ranked_ids=[x['expert_id'] for x in final['ranking']],unresolved=final['unresolved_experts'],decisions=decisions)
        rows.append(dict(id=q['id'],group_id=q['group_id'],arms=arms));print('Completed',n+1,flush=True)
    catalog.db.close();(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    stats={arm:dict(mean_seconds=statistics.mean(r['arms'][arm]['seconds'] for r in rows),valid_assessments=sum(r['arms'][arm]['valid'] for r in rows),partial_queries=sum(r['arms'][arm]['status']!='provisional' for r in rows),errors=dict(Counter(e for r in rows for e in r['arms'][arm]['errors']))) for arm in ('sequential','concurrent')}
    paired=[]
    for row in rows:
        a=row['arms']['sequential']['decisions'];b=row['arms']['concurrent']['decisions']
        if set(a)!=set(b):raise ValueError('Changed candidate set')
        for eid in a:
            if a[eid]['status']=='structurally_valid' and b[eid]['status']=='structurally_valid':paired.append((a[eid],b[eid]))
    result=dict(queries=len(rows),attempts_per_arm=3*len(rows),jointly_valid_candidates=len(paired),same_level=sum(a['level']==b['level'] for a,b in paired),same_citations=sum(a['citations']==b['citations'] for a,b in paired),same_level_and_citations=sum(a['level']==b['level'] and a['citations']==b['citations'] for a,b in paired),level_pairs={str(k):v for k,v in Counter((a['level'],b['level']) for a,b in paired).items()},stats=stats,paired_concurrent_minus_sequential=paired_interval([r['arms']['concurrent']['seconds']-r['arms']['sequential']['seconds'] for r in rows]),
        same_final_rankings=sum(r['arms']['sequential']['ranked_ids']==r['arms']['concurrent']['ranked_ids'] for r in rows),catalog_init_seconds=catalog_init,
        protocol_sha256=sha(output/'protocol.json'),private_results_sha256=sha(output/'private-results.json'),
        scope=protocol['scope']+' Single alternating observation per query; elapsed time includes assessment requests and source reads but not actual query encoding, initial index retrieval, planner API, initialization or competing traffic. Independent model calls can differ; outcomes and failures cannot be interpreted as expert correctness. No retries; bounded concurrency may trigger rate limits. Six queries are insufficient for production claims.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
