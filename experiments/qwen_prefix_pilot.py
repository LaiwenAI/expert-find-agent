"""Paired prefix-cache pilot, fixed first six protocol queries per dataset."""
import hashlib
import json
import random
import time
from pathlib import Path
from expert_search.qwen_prefix_cache import PrefixCachedReranker
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.cof_crossencoder import validate_sources


def run(fp32=False):
    import mlx.core as mx
    mx.set_memory_limit(4*1024**3)
    mx.set_cache_limit(128*1024**2)
    name = 'qwen-prefix-fp32-pilot' if fp32 else 'qwen-prefix-pilot'
    output = Path('artifacts')/(name+'-v1')
    output.mkdir(parents=True,exist_ok=True)
    spec = dict(queries_per_dataset=6,repetitions=2,seed=20261118,
                selection='First six original protocol query IDs, no outcome selection',
                code_sha256=hashlib.sha256(Path('src/expert_search/qwen_prefix_cache.py').read_bytes()).hexdigest())
    if fp32:
        spec['arithmetic'] = 'FP32 model and answer head'
    marker = output/'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError('Changed pilot protocol')
    marker.write_text(json.dumps(spec,indent=2))
    model = PrefixCachedReranker()
    if fp32:
        model.model.apply(lambda value: value.astype(mx.float32))
        model.weight = model.model.model.embed_tokens.weight
        model.answer_weight = model.weight[mx.array(model.answer_ids)].astype(mx.float32)
        mx.eval(model.model.parameters(), model.answer_weight)
    rng = random.Random(spec['seed'])
    rows = []
    for dataset in ('NIPS','SIGIR'):
        root,papers,reviewers,labels = load(dataset)
        inputs = root/'packing-shared-v2'
        protocol = json.loads((inputs/'protocol.json').read_text())
        for qid in protocol['all_query_ids'][:6]:
            payload = json.loads((inputs/f'{qid}-similarity-input.json').read_text())
            aliases = json.loads((inputs/f'{qid}-similarity.json').read_text())['aliases']
            candidates = validate_sources(payload,aliases,papers,reviewers)
            documents = ['\n\n'.join(e['text'] for e in c['evidence']) for c in candidates]
            for repetition in range(2):
                path = output/f'{dataset}-{qid}-{repetition}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                order = ['full','cached']
                rng.shuffle(order)
                results,timings = {},{}
                for method in order:
                    start = time.perf_counter()
                    results[method] = (model.score(payload['target_paper'],documents) if method=='full'
                                       else model.score_cached(payload['target_paper'],documents))
                    timings[method] = time.perf_counter()-start
                rankings = {method: sorted(range(len(aliases)),key=lambda i:(-result['scores'][i],aliases[i]))
                            for method,result in results.items()}
                row = dict(dataset=dataset,query_id=qid,repetition=repetition,order=order,seconds=timings,
                           maximum_probability_difference=max(abs(a-b) for a,b in zip(results['full']['scores'],results['cached']['scores'])),
                           top5_identical=rankings['full'][:5]==rankings['cached'][:5],
                           common_prefix_tokens=results['cached']['common_prefix_tokens'],
                           metrics={method:{task:precision_metrics([aliases[i] for i in rank],truth[qid])
                                            for task,truth in labels.items()} for method,rank in rankings.items()})
                path.write_text(json.dumps(row,indent=2))
                rows.append(row)
                print(dataset,qid,repetition,timings,flush=True)
    report = dict(protocol=spec,trials=len(rows),rows=rows)
    (Path('docs/exploration')/(name+'.json')).write_text(json.dumps(report,indent=2))
    print('Complete',len(rows),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32',action='store_true')
    run(parser.parse_args().fp32)
