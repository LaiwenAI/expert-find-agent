"""Verify observed signature fields; abstract functions remain retrieval hints."""
import hashlib
import json
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend


def run():
    parent=Path('artifacts/expert-method-signatures-v1');out=Path('artifacts/method-signature-audit-v1');out.mkdir(exist_ok=True)
    files=sorted(parent.glob('CN_*.json'))
    protocol=dict(version=1,parents={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        policy='Judge observed input/operation/output/object against original cited source spans. Exclude unspecified fields and abstract routing hints. Same-model-family diagnostic, not human gold.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed signature audit inputs')
    marker.write_text(json.dumps(protocol,indent=2));backend=DecisionsBackend();rows=[]
    for path in files:
        signature=json.loads(path.read_text());tree=json.loads((Path('artifacts/expert-semantic-tree-v1')/path.name).read_text());nodes={n['id']:n for n in tree['nodes']}
        if signature['status']!='structurally_valid':continue
        target=out/path.name
        if target.exists():row=json.loads(target.read_text())
        else:
            pairs=[]
            for s in signature['signatures']:
                node=nodes[s['node_id']];quotes=[]
                for c in node['citations']:
                    source=tree['sources'][c['evidence_id']]
                    if source['expert_id']!=tree['expert_id'] or source['text'][c['start']:c['end']]!=c['quote']:raise ValueError('Changed source span')
                    quotes.append(c['quote'])
                observed={k:s[k] for k in ('observed_input','observed_operation','observed_output','observed_object') if s[k].strip().casefold()!='unspecified'}
                pairs.append(dict(id=s['node_id'],constraint=json.dumps(observed,ensure_ascii=False),text=json.dumps(quotes,ensure_ascii=False)))
            row=dict(expert_id=tree['expert_id'],status='unresolved')
            try:row.update(status='valid',response=backend.judge(pairs))
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(tree['expert_id'],row['status'],flush=True)
    judgments=[j for r in rows if r['status']=='valid' for j in r['response']['judgments']]
    report=dict(experts=len(rows),valid_experts=sum(r['status']=='valid' for r in rows),signatures=len(judgments),labels={label:sum(j['label']==label for j in judgments) for label in ('supported','unknown','contradicted')},warning=protocol['policy'])
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
