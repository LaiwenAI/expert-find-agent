"""Replay frozen vector witnesses against targeted and full-owner catalog reads."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from expert_search.expert_catalog import ExpertCatalog
from expert_search.statistics import clustered_paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    prior=Path('artifacts/bilingual-retrieval-v1');summary=json.loads((prior/'summary.json').read_text());evaluation=prior/'evaluation.jsonl'
    if sha(evaluation)!=summary['evaluation_sha256'] or sha(prior/'protocol.json')!=summary['protocol_sha256']:raise ValueError('Changed frozen retrieval')
    rows=[json.loads(x) for x in evaluation.read_text().splitlines()]
    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Changed query cohort')
    chosen=sorted(random.Random(20261124).sample(range(len(rows)),200));chosen_rows=[rows[i] for i in chosen]
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(queries=200,seed=20261124,selection='Fixed-seed sample from previously explored 1000 construction parents; Chinese query view only.',
        source_evaluation_sha256=sha(evaluation),source_protocol_sha256=sha(prior/'protocol.json'),script_sha256=sha(__file__),catalog_sha256=sha('src/expert_search/expert_catalog.py'),grounded_pipeline_sha256=sha('src/expert_search/grounded_pipeline.py'),
        order='Alternating by query and repeat; one warm-up, three measured repeats.',scope='Frozen top-ten experts and first witness; no relevance, semantic support, identity or model reasoning evaluation.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');start=time.perf_counter();catalog=ExpertCatalog();init=time.perf_counter()-start;cases=[]
    for i,row in enumerate(chosen_rows):
        retrieval=row['languages']['zh']['exact'];expert_ids=retrieval['ranking'];source_ids=retrieval['top_source_ids']
        if len(expert_ids)!=10 or len(source_ids)!=10 or len(set(source_ids))!=10:raise ValueError('Changed top-ten witness inventory')
        seconds={'all_owner':[],'targeted':[]};baseline=None;target=None
        for repeat in range(-1,3):
            outputs={}
            for arm in (('all_owner','targeted') if (i+repeat)%2==0 else ('targeted','all_owner')):
                start=time.perf_counter();outputs[arm]=catalog.for_experts(expert_ids) if arm=='all_owner' else catalog.for_evidence(source_ids);elapsed=time.perf_counter()-start
                if repeat>=0:seconds[arm].append(elapsed)
            all_map={r['id']:r for r in outputs['all_owner']};target_map={r['id']:r for r in outputs['targeted']}
            if len(target_map)!=len(source_ids) or set(target_map)!=set(source_ids) or any(target_map[s]!=all_map.get(s) or target_map[s]['expert_id']!=e for e,s in zip(expert_ids,source_ids)):
                raise ValueError('Changed source content or ownership')
            current=([all_map[s] for s in source_ids],[target_map[s] for s in source_ids])
            if baseline is None:baseline=current
            elif current!=baseline:raise ValueError('Changed snapshot during replay')
            target=len(outputs['all_owner'])
        cases.append(dict(id=row['id'],group_id=row['group_id'],all_owner_records=target,target_records=len(source_ids),seconds=seconds))
    catalog.db.close();(output/'private-results.json').write_text(json.dumps(cases,indent=2)+'\n')
    means={a:[statistics.mean(r['seconds'][a]) for r in cases] for a in ('all_owner','targeted')}
    interval=clustered_paired_interval([a-b for a,b in zip(means['targeted'],means['all_owner'])],[r['group_id'] for r in cases]);interval['sampling_unit']='whole company group; query-weighted mean'
    report=dict(queries=len(cases),matched_complete_selected_sources=len(cases)*8,initialization_seconds=init,mean_records_read={a:statistics.mean(r['all_owner_records' if a=='all_owner' else 'target_records'] for r in cases) for a in means},
        mean_seconds={a:statistics.mean(means[a]) for a in means},targeted_faster_query_means=sum(a<b for a,b in zip(means['targeted'],means['all_owner'])),paired_seconds=interval,
        protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'private-results.json'),scope=protocol['scope']+' Cache-warm single-machine file reads; three dependent repeats per query. Initialization, retrieval, encoding, model inference and concurrency are excluded. The targeted path retains missing or foreign witnesses as unresolved in the pipeline. Group bootstrap is unadjusted; no production speed guarantee.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
