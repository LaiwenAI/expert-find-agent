"""Ordered top-k bound replay over frozen equal-weight fusion scores."""
import json
from pathlib import Path
from expert_search.cof_benchmark import load


def replay(dataset, version):
    root, _, _, _ = load(dataset)
    qids = json.loads((root/'packing-shared-v2/protocol.json').read_text())['all_query_ids']
    rows = []
    for qid in qids:
        result = json.loads((root/f'qwen-reranker-mlx-v{version}'/f'{qid}.json').read_text())
        aliases = json.loads((root/'packing-shared-v2'/f'{qid}-similarity.json').read_text())['aliases']
        baseline = json.loads((root/'language-models-v2'/f'query-{qid}.json').read_text())
        order = [e for e in baseline['full_rankings']['model1-document'] if e in set(aliases)]
        if set(order) != set(aliases):
            raise ValueError('Candidate pool mismatch')
        prior = {e: 1-i/(len(order)-1) if len(order)>1 else 1. for i,e in enumerate(order)}
        slow = {aliases[s['id']]: s['score'] for s in result['scores']}
        complete = {e: .5*(prior[e]+slow[e]) for e in order}
        k = min(5,len(order))
        expected = sorted(order,key=lambda e:(-complete[e],e))[:k]
        known = {}
        for eid in order:
            known[eid] = complete[eid]
            leaders = sorted(known,key=lambda e:(-known[e],e))[:k]
            if len(leaders)<k:
                continue
            cutoff = (-known[leaders[-1]],leaders[-1])
            possible = [e for e in order if e not in known and (-.5*(prior[e]+1.),e)<cutoff]
            if not possible:
                break
        if leaders != expected:
            raise ValueError('Ordered top-k differs from complete reference')
        rows.append(dict(query_id=qid,candidates=len(order),evaluated=len(known)))
    return dict(dataset=dataset,queries=len(rows),rows=rows,
        evaluated=sum(r['evaluated'] for r in rows),total=sum(r['candidates'] for r in rows),
        exact_ordered_top5=True,
        limitations='Replay of frozen retrospective equal-weight fusion, candidate-at-a-time bound. Actual GPU batching, token lengths and repeated inference can change cost and numerical scores. No measured online speedup.')


if __name__=='__main__':
    report=[replay('NIPS',1),replay('SIGIR',2)]
    Path('docs/exploration/qwen-pruning-replay.json').write_text(json.dumps(report,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in report],indent=2))
