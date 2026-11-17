"""Compare native choices and previous Chat role hints without treating either as gold."""
import json
import statistics
from collections import Counter
from pathlib import Path

from expert_search.owned_role_windows import segment_source,source_key
from .owned_role_decision_pilot import sha


def labels(record,n):
    result={}
    for role,ids in record['roles'].items():
        for i in ids:
            if i in result:raise ValueError('Conflicting normalized role')
            result[i]=role
    if set(result)!=set(range(n)):raise ValueError('Incomplete normalized roles')
    return result


def main():
    root=Path('artifacts/owned-role-decision-pilot-v1');parent=Path('artifacts/owned-role-paper-pilot-v1')
    p=json.loads((root/'protocol.json').read_text());raw=(parent/'sources.json').read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=p['source_sha256'] or sha(parent/'protocol.json')!=p['previous_protocol_sha256']:raise ValueError('Changed source or prior protocol')
    for path,digest in p['code_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed code')
    sources=json.loads(raw)
    if len(sources)!=p['sources']:raise ValueError('Changed source count')
    statuses={arm:Counter() for arm in ('chat','decision')};counts={arm:Counter() for arm in statuses};seconds={arm:[] for arm in statuses}
    usages={arm:[] for arm in statuses};token_counts={arm:[] for arm in statuses};matched=agree=0;matrix=Counter();eligible_sources=0
    for i,source in enumerate(sources,1):
        name=f'source-{i:03d}.json'
        if sha(parent/name)!=p['previous_attempt_sha256'][name]:raise ValueError('Changed prior attempt')
        if not (root/name).exists():raise ValueError('Incomplete run')
        records={'chat':json.loads((parent/name).read_text()),'decision':json.loads((root/name).read_text())}
        n=len(segment_source(source,p['width']));key=source_key(source,p['width']);per={}
        for arm,r in records.items():
            if r['source_key']!=key:raise ValueError('Changed source binding')
            statuses[arm][r['status']]+=1;seconds[arm].append(r['seconds']);usages[arm].append(r.get('usage_present') is True)
            if r.get('usage_present') is True:
                usage=r['usage'];names=('prompt_tokens','completion_tokens') if arm=='chat' else ('input_tokens','output_tokens')
                if any(type(usage.get(k))!=int or usage[k]<0 for k in names):raise ValueError('Invalid token accounting')
                token_counts[arm].append(tuple(usage[k] for k in names))
            if r['status']=='normalized_role_hints':
                if r['role_record']['source_key']!=key:raise ValueError('Changed role binding')
                per[arm]=labels(r['role_record'],n);counts[arm].update(per[arm].values())
        if len(per)==2:
            eligible_sources+=1
            for j in range(n):
                a,b=per['chat'][j],per['decision'][j];matched+=1;agree+=a==b;matrix[a,b]+=1
    report={'sources':len(sources),'common_valid_sources':eligible_sources,'matched_windows':matched,'same_labels':agree,
            'roles_by_arm':{a:dict(v) for a,v in counts.items()},'statuses_by_arm':{a:dict(v) for a,v in statuses.items()},
            'mean_seconds_by_arm':{a:statistics.mean(v) for a,v in seconds.items()},'complete_usage_by_arm':{a:all(v) for a,v in usages.items()},
            'mean_input_tokens_by_arm':{a:statistics.mean(v[0] for v in token_counts[a]) if all(usages[a]) else None for a in statuses},
            'mean_output_tokens_by_arm':{a:statistics.mean(v[1] for v in token_counts[a]) if all(usages[a]) else None for a in statuses},
            'label_pairs':[{'chat':a,'decision':b,'count':n} for (a,b),n in sorted(matrix.items())],
            'scope':'Sequential post-hoc comparison of different prompts and output interfaces on the same paper views. Agreement is not correctness; native single choices prevent structural multi-role conflicts but can still be wrong or uncertain. No random order, human gold, retrieval-quality or latency-causality claim.'}
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
