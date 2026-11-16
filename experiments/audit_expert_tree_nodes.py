"""Audit offline directory summaries against only their attributed citations."""
import hashlib
import inspect
import json
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend


def run():
    parent=Path('artifacts/expert-semantic-tree-v1')
    if not (parent/'summary.json').exists():raise ValueError('Wait for complete tree construction')
    files=sorted(parent.glob('CN_*.json'));out=Path('artifacts/expert-tree-node-audit-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,parent_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},backend_sha256=hashlib.sha256(inspect.getsource(DecisionsBackend).encode()).hexdigest(),
        policy='Whole node title/problem/methods/objects judged against cited original spans only. Same-family diagnostic. Non-supported node cannot serve as verified summary; original source fallback retained.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed node audit protocol')
    marker.write_text(json.dumps(protocol,indent=2));backend=DecisionsBackend();rows=[]
    for path in files:
        tree=json.loads(path.read_text());target=out/path.name
        if tree['status']!='structurally_valid' or not tree['nodes']:continue
        if target.exists():row=json.loads(target.read_text())
        else:
            row=dict(expert_id=tree['expert_id'],status='unresolved')
            pairs=[]
            for node in tree['nodes']:
                texts=[]
                for c in node['citations']:
                    source=tree['sources'][c['evidence_id']]
                    if source['expert_id']!=tree['expert_id'] or source['text'][c['start']:c['end']]!=c['quote']:raise ValueError('Changed source citation')
                    texts.append(dict(evidence_id=c['evidence_id'],quote=c['quote']))
                statement={k:node[k] for k in ('title','research_problem','methods','objects')}
                pairs.append(dict(id=node['id'],constraint=json.dumps(statement,ensure_ascii=False),text=json.dumps(texts,ensure_ascii=False)))
            try:row.update(status='valid',response=backend.judge(pairs))
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,ensure_ascii=False,indent=2));temporary.replace(target)
        rows.append(row);print(tree['expert_id'],row['status'],flush=True)
    judgments=[j for r in rows if r['status']=='valid' for j in r['response']['judgments']]
    report=dict(experts_judged=len(rows),valid_experts=sum(r['status']=='valid' for r in rows),nodes_judged=len(judgments),
        labels={label:sum(j['label']==label for j in judgments) for label in ('supported','contradicted','unknown')},
        warning='Cited-span entailment diagnostic, same provider family; no human gold or guarantee of routing relevance. Original sources retained for all experts.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
