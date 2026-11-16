"""Compare candidate-only catalog access with full source scans."""
import json
import time
from pathlib import Path
from expert_search.corpus import records
from expert_search.expert_catalog import ExpertCatalog


def run():
    jobs=[json.loads(l) for l in Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines()]
    ids=sorted({j['expert_id'] for j in jobs});wanted=set(ids);start=time.perf_counter();catalog=ExpertCatalog();init=time.perf_counter()-start
    observations=[]
    for repetition in range(3):
        order=['catalog','scan'] if repetition%2 else ['scan','catalog'];outputs={};seconds={}
        for mode in order:
            start=time.perf_counter()
            outputs[mode]=catalog.for_experts(ids) if mode=='catalog' else [r for _,r in records('artifacts/corpus/evidence.jsonl') if r['expert_id'] in wanted]
            seconds[mode]=time.perf_counter()-start
        if outputs['scan']!=outputs['catalog']:raise AssertionError('Catalog changed evidence content/order')
        observations.append(seconds)
    report=dict(experts=len(ids),returned_sources=len(outputs['catalog']),initialization_seconds=init,paired_runs=observations,exact_content_and_order=True,
        scope='Candidate-owned source access only, not end-to-end retrieval or evidence reasoning. Engineering prerequisite for semantic trees.')
    Path('artifacts/expert-catalog-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
