"""Recompute fidelity/timing aggregates and verify original source ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def run(directory):
    root=Path(directory);protocol=json.loads((root/'protocol.json').read_text())
    base=Path('artifacts/global-owned-routing-v1/protocol.json')
    source_spec=protocol
    if 'parent_protocol_sha256' in protocol:
        if hashlib.sha256(base.read_bytes()).hexdigest()!=protocol['parent_protocol_sha256']:raise ValueError('Changed parent protocol')
        source_spec=json.loads(base.read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_spec['source_sha256']:raise ValueError('Changed source snapshot')
    inventory=[json.loads(s) for s in source.read_text().splitlines()]
    owners={r['id']:r['expert_id'] for r in inventory}
    if len(owners)!=len(inventory):raise ValueError('Duplicate source IDs')
    rows=json.loads((root/'evaluation.json').read_text());report={}
    with np.load(base.parent/'query-vectors.npz',allow_pickle=False) as f:expected=list(f['ids'])
    if [r['id'] for r in rows]!=expected:raise ValueError('Changed query order')
    witness_checks=0
    for row in rows:
        outputs=row.get('outputs',row.get('results'));truth=outputs['exact_full']['hits'];exact={h['id']:h for h in truth};truth_ids=[h['id'] for h in truth]
        for method,result in outputs.items():
            hits=result['hits']
            if hits!=sorted(hits,key=lambda h:(-h['score'],h['id'])):raise ValueError('Unsorted expert scores')
            if len({h['id'] for h in hits})!=len(hits):raise ValueError('Duplicate experts')
            for h in hits:
                evidence=h['evidence']
                if not evidence or h['score']!=evidence[0]['score']:raise ValueError('Missing max witness')
                if evidence!=sorted(evidence,key=lambda s:(-s['score'],s['id'])):raise ValueError('Unsorted source witnesses')
                for s in evidence:
                    if owners.get(s['id'])!=h['id']:raise ValueError('Foreign source witness')
                if method!='exact_full' and h['id'] in exact:
                    if evidence!=exact[h['id']]['evidence'] or h['score']!=exact[h['id']]['score']:raise ValueError('Common expert rescoring differs')
                    witness_checks+=1
            ids=[h['id'] for h in hits];timing=row.get('timings',row.get('timing'))[method]
            if len(timing['samples'])!=5 or float(np.median(timing['samples']))!=timing['median']:raise ValueError('Timing aggregation differs')
            report.setdefault(method,[]).append(dict(ordered=ids==truth_ids,retained=len(set(ids)&set(truth_ids))/len(truth_ids),seconds=timing['median']))
    computed={m:dict(ordered_top10_equal=sum(r['ordered'] for r in values),mean_exact_top10_retained=float(np.mean([r['retained'] for r in values])),mean_median_seconds=float(np.mean([r['seconds'] for r in values]))) for m,values in report.items()}
    summary=json.loads((root/'summary.json').read_text())
    if 'methods' in summary:
        if computed!=summary['methods']:raise ValueError('Summary metrics differ')
    else:
        for name in ['ordered_top10_equal','mean_exact_top10_retained']:
            if computed['owned_ann'][name]!=summary[name]:raise ValueError('Summary fidelity differs')
        if {m:r['mean_median_seconds'] for m,r in computed.items()}!=summary['mean_median_seconds']:raise ValueError('Summary timings differ')
    result=dict(queries=len(rows),all_source_owners_checked=True,common_expert_exact_witness_checks=witness_checks,computed=computed,
                files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'protocol.json',root/'evaluation.json',root/'summary.json']},
                scope='Saved-output integrity and exact-vector ranking fidelity only. No source semantic support, expert fit, ANN certificate or hardware speed guarantee.')
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='artifacts/global-owned-routing-v1');run(parser.parse_args().directory)
