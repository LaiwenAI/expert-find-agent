"""Freeze a broader paper-backed construction cohort before model judgments."""
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
from expert_search.corpus import records


def run():
    parent=Path('artifacts/demand-topic-overlap-v1/candidates.jsonl');rng=random.Random(20261117)
    excluded_demands={r['id'] for _,r in records('artifacts/demands/queries.jsonl')}
    excluded_demands.update(p.stem for p in Path('artifacts/topic-overlap-audit-v1').glob('demand-*.json'))
    excluded_experts={r['constructing_expert_id'] for _,r in records('artifacts/dataset-release/scoped-capabilities-audited/pairs.jsonl')}
    excluded_experts.update(json.loads(p.read_text())['expert_id'] for p in Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json'))
    counts=Counter(r['expert_id'] for _,r in records('artifacts/corpus/evidence.jsonl') if r['kind']=='paper' and r['text'].strip())
    groups=defaultdict(list)
    for _,row in records(parent):
        if row['demand_id'] not in excluded_demands and row['expert_id'] not in excluded_experts and counts[row['expert_id']]>=6:groups[row['demand_id']].append(row)
    order=sorted(groups);rng.shuffle(order);chosen=[];used=set()
    for did in order:
        candidates=sorted(groups[did],key=lambda r:r['expert_id']);rng.shuffle(candidates)
        candidate=next((r for r in candidates if r['expert_id'] not in used),None)
        if candidate is not None:chosen.append(candidate);used.add(candidate['expert_id'])
        if len(chosen)==20:break
    out=Path('artifacts/expanded-topic-overlap-v1');out.mkdir(exist_ok=True)
    body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)
    protocol=dict(version=1,seed=20261117,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),eligible_demands=len(groups),selected_demands=len(chosen),
        excluded_demands=sorted(excluded_demands),excluded_experts=sorted(excluded_experts),selection='Uniform shuffled demand order; one shuffled eligible expert per demand; distinct experts, >=6 nonempty papers for matched source controls.',
        candidates_sha256=hashlib.sha256(body.encode()).hexdigest(),warning='Topic-overlap and paper-availability conditioned new construction cohort, not population representative. Prior demand/expert IDs excluded; exact/near source-text overlap still needs grouped audit. No model outcomes used.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed expanded cohort')
    marker.write_text(json.dumps(protocol,indent=2));(out/'candidates.jsonl').write_text(body);print('Frozen',len(chosen),'new demands and experts from',len(groups),'eligible demands')


if __name__=='__main__':run()
