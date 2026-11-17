"""Replay deterministic per-paper role reuse using saved test sentence vectors."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from expert_search.evidencebench_data import aspect_coverage
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.source_roles import ROLES,normalize_roles,prioritize_roles
from evidencebench_pilot import TASK


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/evidencebench-role-test-v1');data_path=Path('artifacts/public/evidencebench/test.json')
    data=json.loads(data_path.read_text());protocol=json.loads((root/'protocol.json').read_text())
    if digest(data_path)!=protocol['dataset_sha256']:raise ValueError('Changed dataset')
    for path,expected in protocol['code_sha256'].items():
        if digest(Path(path))!=expected:raise ValueError('Changed original evaluator: '+path)
    historic=json.loads((root/'evaluation.json').read_text());historic={r['id']:r for r in historic}
    if historic.keys()!=data.keys():raise ValueError('Incomplete historical output')
    groups=defaultdict(list)
    for key,row in data.items():groups[row['paper_id']].append(key)
    source_ids={}
    for group in groups.values():
        snapshots={hashlib.sha256(json.dumps(data[key]['paper_as_candidate_pool'],ensure_ascii=False).encode()).hexdigest() for key in group}
        if len(snapshots)!=1:raise ValueError('Same paper has different sentence snapshots')
        for key in group:source_ids[key]=min(group)
    roles={};durations={}
    for key,row in data.items():
        record=json.loads((root/'roles'/(key+'.json')).read_text());durations[key]=record['offline_seconds']
        if record['status']=='normalized':
            value=json.loads(record['response']['choices'][0]['message']['content'])
            normalized=normalize_roles(value,len(row['paper_as_candidate_pool']))['roles']
            if normalized!=record['roles']:raise ValueError('Changed role normalization')
            roles[key]=normalized
        else:roles[key]={role:[] for role in ROLES}
    model=QwenTorchEmbedding(max_length=512,batch_size=8);results=[]
    for key,row in data.items():
        vectors=np.load(root/'vectors'/(key+'.npy'),allow_pickle=False)
        if vectors.shape!=(len(row['paper_as_candidate_pool']),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid vectors')
        q=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0]
        scores=vectors@q;ranking=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
        original=prioritize_roles(ranking,roles[key])[:10];cached=prioritize_roles(ranking,roles[source_ids[key]])[:10]
        if ranking[:10]!=historic[key]['baseline_selected'] or original!=historic[key]['selected']:
            raise ValueError('Recomputed ordering differs from historical run')
        old=aspect_coverage(row,original)['aspect_recall'];new=aspect_coverage(row,cached)['aspect_recall']
        results.append({'id':key,'canonical_id':source_ids[key],'selection_changed':original!=cached,
                        'original_aspect_recall':old,'cached_aspect_recall':new})
    out=Path('artifacts/evidencebench-role-reuse-replay-v1');out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    report={'queries':len(results),'unique_papers':len(groups),'changed_selections':sum(r['selection_changed'] for r in results),
            'mean_original_aspect_recall':float(np.mean([r['original_aspect_recall'] for r in results])),
            'mean_cached_aspect_recall':float(np.mean([r['cached_aspect_recall'] for r in results])),
            'original_sum_request_seconds':sum(durations.values()),
            'canonical_sum_request_seconds':sum(durations[min(group)] for group in groups.values()),
            'dataset_sha256':digest(data_path),'original_protocol_sha256':digest(root/'protocol.json'),'original_evaluation_sha256':digest(root/'evaluation.json'),
            'scope':'Post-hoc replay of one lexicographically selected existing source-only compilation per paper. No new model calls; historical vector ranking and selections must reproduce exactly. Offline request sums are not elapsed wall time. Not independent validation, enterprise relevance, or a causal latency comparison.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
