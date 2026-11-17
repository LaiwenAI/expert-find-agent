#!/bin/bash
#SBATCH --job-name=evidence-sensitivity
#SBATCH --partition=gpu
#SBATCH --nodelist=ws5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-sensitivity-%j.out
#SBATCH --error=/media/Ws5_4TB/zdx/expert-find-agent/logs/evidence-sensitivity-%j.err
set -euo pipefail
cd /media/Ws5_4TB/zdx/expert-find-agent
export PYTHONPATH="$PWD/src:$PWD/experiments"
export HF_HOME="$PWD/cache/huggingface"
export TMPDIR="$PWD/tmp"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
venv/bin/python - <<'MODEL_CHECK'
import hashlib,json,torch
from pathlib import Path
root=Path('artifacts/models/qwen3-embedding-0.6b')
for name,expected in json.loads((root/'checkpoint-provenance.json').read_text())['files'].items():
    with (root/name).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Changed checkpoint: '+name)
if not torch.cuda.is_available():raise RuntimeError('No allocated GPU')
MODEL_CHECK
venv/bin/python - <<'REPLAY_SOURCE'
"""Post-hoc exhaustive reuse sensitivity over saved repeated-source compilations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from expert_search.evidencebench_data import aspect_coverage
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.source_roles import normalize_roles,prioritize_roles
from evidencebench_pilot import TASK


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/evidencebench-role-test-v1');data_path=Path('artifacts/public/evidencebench/test.json')
    data=json.loads(data_path.read_text());protocol=json.loads((root/'protocol.json').read_text())
    if sha(data_path)!=protocol['dataset_sha256']:raise ValueError('Changed dataset')
    for path,expected in protocol['code_sha256'].items():
        if sha(Path(path))!=expected:raise ValueError('Changed original evaluator: '+path)
    history={r['id']:r for r in json.loads((root/'evaluation.json').read_text())}
    if history.keys()!=data.keys():raise ValueError('Incomplete historical results')
    groups=defaultdict(list)
    for key,row in data.items():groups[row['paper_id']].append(key)
    repeated=[sorted(group) for group in groups.values() if len(group)>1]
    model=QwenTorchEmbedding(max_length=512,batch_size=8);deltas=[];changed_by_choice=[];comparisons=0;group_rows=[]
    for ids in repeated:
        snapshots={hashlib.sha256(json.dumps(data[key]['paper_as_candidate_pool'],ensure_ascii=False).encode()).hexdigest() for key in ids}
        if len(snapshots)!=1:raise ValueError('Different source snapshots')
        roles={};rankings={}
        for key in ids:
            row=data[key];record=json.loads((root/'roles'/(key+'.json')).read_text())
            if record['status']!='normalized':raise ValueError('Repeated group includes invalid compilation')
            roles[key]=normalize_roles(json.loads(record['response']['choices'][0]['message']['content']),len(row['paper_as_candidate_pool']))['roles']
            if roles[key]!=record['roles']:raise ValueError('Changed role normalization')
            vectors=np.load(root/'vectors'/(key+'.npy'),allow_pickle=False)
            if vectors.shape!=(len(row['paper_as_candidate_pool']),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid vectors')
            query=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0]
            scores=vectors@query;ranking=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i));rankings[key]=ranking
            if ranking[:10]!=history[key]['baseline_selected'] or prioritize_roles(ranking,roles[key])[:10]!=history[key]['selected']:
                raise ValueError('Historical ranking not reconstructed')
        baseline=sum(history[key]['metrics']['aspect_recall'] for key in ids);choices=[]
        for canonical in ids:
            scores=[];changed=0
            for key in ids:
                chosen=prioritize_roles(rankings[key],roles[canonical])[:10]
                scores.append(aspect_coverage(data[key],chosen)['aspect_recall'])
                changed+=chosen!=history[key]['selected'];comparisons+=1
            choices.append({'delta_sum':sum(scores)-baseline,'changed_selections':changed})
        deltas.append((min(c['delta_sum'] for c in choices),max(c['delta_sum'] for c in choices)))
        changed_by_choice.extend(c['changed_selections'] for c in choices)
        group_rows.append({'queries':len(ids),'choices':len(choices),'minimum_delta_sum':deltas[-1][0],'maximum_delta_sum':deltas[-1][1]})
    original=float(np.mean([r['metrics']['aspect_recall'] for r in history.values()]));n=len(history)
    result={'queries':n,'repeated_paper_groups':len(repeated),'repeated_queries':sum(map(len,repeated)),
            'counterfactual_query_compilation_pairs':comparisons,'original_mean':original,
            'min_mean_over_existing_choices':original+sum(x[0] for x in deltas)/n,
            'max_mean_over_existing_choices':original+sum(x[1] for x in deltas)/n,
            'maximum_changed_lists_in_one_group_choice':max(changed_by_choice,default=0),'group_results':group_rows,
            'dataset_sha256':sha(data_path),'original_protocol_sha256':sha(root/'protocol.json'),'original_evaluation_sha256':sha(root/'evaluation.json'),
            'scope':'Post-hoc exhaustive choices among existing compilations only, shared within each paper group. All historical rankings reproduced; no new LLM calls. Bounds describe this finite replay, not uncertainty, sentence semantic equivalence, new draws, enterprise relevance, or full-system speed.'}
    out=Path('artifacts/evidencebench-role-reuse-sensitivity-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

REPLAY_SOURCE
