"""Conservative connected-component splits for paired provenance datasets."""
import hashlib
import json
import random
import unicodedata
from pathlib import Path
from .corpus import records


def resources(row):
    result = set()
    for kind,key in (('demand','parent_demand_id'),('expert','constructing_expert_id')):
        value=row.get(key)
        if not isinstance(value,str) or not value:
            raise ValueError('Missing grouping identifier')
        result.add((kind,value))
    for key in ('query','counterfactual_query'):
        text=row.get(key)
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Missing paired query')
        result.add(('query',' '.join(unicodedata.normalize('NFKC',text).casefold().split())))
    if not isinstance(row.get('references'),list) or not row['references']:
        raise ValueError('Missing provenance references')
    for reference in row['references']+row.get('source_snapshots',[]):
        digest=reference.get('source_snapshot_sha256')
        if not isinstance(digest,str) or not digest:
            raise ValueError('Missing source snapshot hash')
        result.add(('source',digest))
    return result


def split_components(rows, seed=42):
    ids=[r.get('id') for r in rows]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Require unique nonempty pair IDs')
    parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    row_resources={r['id']:resources(r) for r in rows}
    for identifier in sorted(ids):
        for resource in sorted(row_resources[identifier]):
            if resource in seen:
                a,b=find(identifier),find(seen[resource])
                parent[max(a,b)]=min(a,b)
            else:
                seen[resource]=identifier
    components={}
    for identifier in sorted(ids):
        components.setdefault(find(identifier),[]).append(identifier)
    groups=list(components.values())
    if len(groups)<3:
        return dict(feasible=False,components=groups,reason='Fewer than three disjoint components; no artificial row-level split')
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group:-len(group))
    names=('train','validation','test')
    weights=(.6,.2,.2)
    selected={name:[] for name in names}
    for number,group in enumerate(groups):
        empty=[name for name in names if not selected[name]]
        options=empty if len(groups)-number==len(empty) else names
        chosen=max(options,key=lambda name:weights[names.index(name)]*len(rows)-len(selected[name]))
        selected[chosen].extend(group)
    assigned_resources={name:set().union(*(row_resources[i] for i in selected[name])) for name in names}
    for index,name in enumerate(names):
        if any(assigned_resources[name]&assigned_resources[other] for other in names[index+1:]):
            raise ValueError('Resource leakage across splits')
    return dict(feasible=True,seed=seed,target_fractions=dict(zip(names,weights)),components=groups,
                assignments={name:sorted(values) for name,values in selected.items()},
                counts={name:len(values) for name,values in selected.items()},largest_component=max(map(len,groups)),
                zero_cross_split_overlap=True,
                grouping='Transitive demand, constructing expert, all input and referenced source snapshot hashes, normalized full base/counterfactual query',
                warning='Small pilot split, not a powered evaluation. Templates remain shared. Near-duplicate sources and semantic query paraphrases are not ruled out. Component sizes can prevent target fractions.')


def run(directory='artifacts/dataset-release/scoped-capabilities-audited'):
    root=Path(directory)
    rows=[r for _,r in records(root/'pairs.jsonl')]
    result=split_components(rows)
    result['pairs_sha256']=hashlib.sha256((root/'pairs.jsonl').read_bytes()).hexdigest()
    if result['feasible']:
        by_id={r['id']:r for r in rows}
        result['file_sha256']={}
        for name,ids in result['assignments'].items():
            target=root/(name+'.jsonl')
            target.write_text(''.join(json.dumps(by_id[i],ensure_ascii=False)+'\n' for i in ids))
            result['file_sha256'][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'splits.json').write_text(json.dumps(result,indent=2))
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['split']='component-disjoint pilot' if result['feasible'] else 'unsplit: fewer than three components'
    manifest['split_counts']=result.get('counts')
    manifest['split_manifest_sha256']=hashlib.sha256((root/'splits.json').read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
