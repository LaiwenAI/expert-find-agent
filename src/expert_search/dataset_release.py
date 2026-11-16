"""Package only full-query-audited silver candidates; never relabel them as gold.
Release metadata omits original source text, which has separate redistribution terms.
"""
import hashlib
import json
from pathlib import Path


def build(output='artifacts/dataset-release'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);accepted=[];rejected=[]
    for p in sorted(Path('artifacts/synthetic-repaired').glob('*.json')):
        r=json.loads(p.read_text());audit=Path('artifacts/synthetic-repaired-query-audit')/p.name
        if r['status']=='rejected' or not audit.exists() or json.loads(audit.read_text())['status']!='silver_candidate':
            rejected.append(p.stem);continue
        verified=json.loads(audit.read_text())['audit']
        counterfactual=dict(r['example']['counterfactual'])
        counterfactual['generator_label']=counterfactual.get('expected_label')
        counterfactual['auditor_label']=verified['counterfactual_label']
        counterfactual['label_provenance']='See generator_label and auditor_label; neither is human gold'
        constraints=[]
        for c in r['example']['constraints']:
            evidence=r['sources'].get(c.get('evidence_id'))
            constraints.append(dict(text=c['text'],label=c['label'],evidence_id=c.get('evidence_id'),
                source_sha256=hashlib.sha256(evidence['text'].encode()).hexdigest() if evidence else None))
        accepted.append(dict(id=p.stem,query=r['example']['query'],expert_id=r['expert_id'],family_id=r['family_id'],constraints=constraints,
            counterfactual=counterfactual,label_quality='model-audited silver',source_artifact=str(p)))
    # Connect repeated experts, families and exact source texts before splitting.
    parent=list(range(len(accepted)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(accepted):
        keys=['expert:'+r['expert_id'],'family:'+r['family_id']]+['source:'+c['source_sha256'] for c in r['constraints'] if c['source_sha256']]
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    groups={}
    for i in range(len(accepted)):groups.setdefault(find(i),[]).append(i)
    for members in groups.values():
        group='|'.join(sorted(accepted[i]['id'] for i in members));bucket=int(hashlib.sha256(group.encode()).hexdigest(),16)%10
        split='test' if bucket==0 else 'dev' if bucket==1 else 'train'
        for i in members:accepted[i]['split']=split
    (out/'silver.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    manifest=dict(examples=len(accepted),rejected=rejected,split_counts={s:sum(r['split']==s for r in accepted) for s in ['train','dev','test']},
        warning='Pilot only. Too small for benchmark claims. Source text is excluded; derived query redistribution still requires release review. No public publishing performed.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':print(json.dumps(build(),indent=2))
