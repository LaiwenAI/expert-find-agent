"""Analyze complete paired native-choice order controls without quality imputation."""
import json
import statistics
from collections import Counter
from pathlib import Path

from expert_search.owned_role_windows import segment_source,source_key
from .owned_role_decision_pilot import sha
from .analyze_owned_role_decision import labels


def main():
    root=Path('artifacts/owned-role-order-pilot-v1');parent=Path('artifacts/owned-role-paper-pilot-v1')
    p=json.loads((root/'protocol.json').read_text())
    if sha(parent/'protocol.json')!=p['parent_protocol_sha256'] or sha(parent/'sources.json')!=p['source_sha256']:raise ValueError('Changed source or protocol')
    for path,digest in p['code_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed code')
    sources=json.loads((parent/'sources.json').read_text())
    if len(sources)!=p['sources']:raise ValueError('Changed source count')
    expected={(i,order) for i in range(len(sources)) for order in ('forward','reverse')}
    schedule=[(r['index'],r['order']) for r in p['schedule']]
    if len(schedule)!=len(expected) or set(schedule)!=expected:raise ValueError('Invalid schedule')
    earlier_root=Path('artifacts/owned-role-decision-pilot-v1');earlier_protocol=json.loads((earlier_root/'protocol.json').read_text())
    if earlier_protocol['source_sha256']!=p['source_sha256']:raise ValueError('Changed earlier source')
    for path,digest in earlier_protocol['code_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Changed earlier decision code')
    statuses=Counter();timing={'forward':[],'reverse':[]};usage={'forward':[],'reverse':[]};matrix=Counter();valid_pairs=0;matched=0;exact_sources=0;models=set()
    same_order_matched=same_order_different=0;earlier_hashes={}
    for i,source in enumerate(sources):
        pair={};key=source_key(source,p['width']);n=len(segment_source(source,p['width']))
        for order in ('forward','reverse'):
            path=root/(f'source-{i+1:03d}-{order}.json')
            if not path.exists():raise ValueError('Incomplete run')
            r=json.loads(path.read_text())
            if (r['source_index'],r['order'],r['source_key'])!=(i,order,key):raise ValueError('Changed source binding')
            statuses[order,r['status']]+=1;timing[order].append(r['seconds']);usage[order].append(r.get('usage_present') is True)
            if r['status']=='normalized_role_hints':
                if r['role_record']['source_key']!=key:raise ValueError('Changed role record')
                pair[order]=labels(r['role_record'],n);models.add(str(r.get('model')))
        if len(pair)==2:
            valid_pairs+=1;exact_sources+=pair['forward']==pair['reverse']
            for j in range(n):
                a,b=pair['forward'][j],pair['reverse'][j];matrix[a,b]+=1;matched+=1
        earlier_path=earlier_root/(f'source-{i+1:03d}.json');earlier_hashes[earlier_path.name]=sha(earlier_path)
        previous=json.loads(earlier_path.read_text())
        if previous['source_key']!=key:raise ValueError('Changed earlier source binding')
        if previous['status']=='normalized_role_hints' and 'forward' in pair:
            old=labels(previous['role_record'],n)
            for j in range(n):
                same_order_matched+=1;same_order_different+=old[j]!=pair['forward'][j]
    output={'sources':len(sources),'common_valid_pairs':valid_pairs,'matched_windows':matched,
            'same_label_windows':sum(n for (a,b),n in matrix.items() if a==b),'identical_sources':exact_sources,
            'statuses':[{'order':a,'status':b,'count':n} for (a,b),n in sorted(statuses.items())],
            'label_pairs':[{'forward':a,'reverse':b,'count':n} for (a,b),n in sorted(matrix.items())],
            'mean_seconds':{a:statistics.mean(v) for a,v in timing.items()},'complete_usage':{a:all(v) for a,v in usage.items()},'models':sorted(models),
            'earlier_forward_comparison':{'matched_windows':same_order_matched,'different_windows':same_order_different,
                                          'protocol_sha256':sha(earlier_root/'protocol.json'),'attempt_sha256':earlier_hashes},
            'scope':'Post-hoc paired choice-order check on 12 reused paper views; one call per order. Earlier same-order calls were sequential and not randomized with this run; identical request structure is reconstructed from frozen code, not recorded wire payload. Differences include model/request variability, not necessarily a causal order effect. Agreement does not establish semantic correctness or evidence attribution.'}
    (root/'analysis.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
