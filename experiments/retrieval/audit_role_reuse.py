"""Audit repeated source-only role compilations on EvidenceBench test data."""
import hashlib
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

from expert_search.source_roles import ROLES,normalize_roles


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/evidencebench-role-test-v1');data_path=Path('artifacts/public/evidencebench/test.json')
    data=json.loads(data_path.read_text());protocol=json.loads((root/'protocol.json').read_text())
    if digest(data_path)!=protocol['dataset_sha256']:raise ValueError('Changed dataset')
    groups=defaultdict(list)
    for key,row in data.items():groups[row['paper_id']].append(key)
    duplicates=[ids for ids in groups.values() if len(ids)>1];comparisons=[];hashes={}
    for ids in duplicates:
        snapshots={hashlib.sha256(json.dumps(data[key]['paper_as_candidate_pool'],ensure_ascii=False).encode()).hexdigest() for key in ids}
        if len(snapshots)!=1:raise ValueError('Repeated paper has different source snapshots')
        roles={}
        for key in ids:
            path=root/'roles'/(key+'.json');hashes[key]=digest(path);record=json.loads(path.read_text())
            if record['status']=='normalized':
                raw=json.loads(record['response']['choices'][0]['message']['content'])
                verified=normalize_roles(raw,len(data[key]['paper_as_candidate_pool']))['roles']
                if verified!=record['roles']:raise ValueError('Saved normalization mismatch')
                roles[key]=verified
        for a,b in itertools.combinations(ids,2):
            if a not in roles or b not in roles:continue
            first,second=roles[a],roles[b]
            x=set().union(*(set(first[r]) for r in ROLES[:3]));y=set().union(*(set(second[r]) for r in ROLES[:3]))
            comparisons.append({'roles_identical':first==second,'preferred_sets_identical':x==y,
                                'preferred_jaccard':len(x&y)/len(x|y) if x|y else 1.})
    report={'repeated_paper_groups':len(duplicates),'queries_in_repeated_groups':sum(map(len,duplicates)),
            'identical_source_snapshot_groups':len(duplicates),'valid_compilation_pairs':len(comparisons),
            'identical_role_pairs':sum(r['roles_identical'] for r in comparisons),
            'identical_preferred_pairs':sum(r['preferred_sets_identical'] for r in comparisons),
            'mean_preferred_jaccard':statistics.mean(r['preferred_jaccard'] for r in comparisons) if comparisons else None,
            'dataset_sha256':digest(data_path),'protocol_sha256':digest(root/'protocol.json'),'role_record_sha256':hashes,
            'scope':'Repeated independent model compilations of identical test paper snapshots; agreement is stability, not semantic accuracy. Groups are few and were inspected after the test. Reusing one compilation would alter the historical protocol and requires separate evaluation. No source text or model outputs are exported.'}
    out=Path('artifacts/evidencebench-role-reuse-audit-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='role_record_sha256'},indent=2))


if __name__=='__main__':main()
