"""Keep observed signature facts separate from functional search hints."""
import hashlib
import json
from pathlib import Path


def run():
    audits=Path('artifacts/method-signature-audit-v1')
    if not (audits/'summary.json').exists():raise ValueError('Signature audit incomplete')
    rows=[];rejected=[]
    for path in sorted(audits.glob('CN_*.json')):
        audit=json.loads(path.read_text());signatures=json.loads((Path('artifacts/expert-method-signatures-v1')/path.name).read_text())
        tree=json.loads((Path('artifacts/expert-semantic-tree-v1')/path.name).read_text());nodes={n['id']:n for n in tree['nodes']}
        labels={j['id']:j['label'] for j in audit.get('response',{}).get('judgments',[])}
        for s in signatures.get('signatures',[]):
            if labels.get(s['node_id'])!='supported':rejected.append(s['node_id']);continue
            node=nodes[s['node_id']]
            rows.append(dict(id=s['node_id'],expert_id=tree['expert_id'],observed={k:s[k] for k in ('observed_input','observed_operation','observed_output','observed_object')},
                original_problem=node['research_problem'],citations=node['citations'],
                search_hint=dict(text=s['abstract_search_function'],role='retrieval_hypothesis_not_evidence',assumptions=s['transfer_assumptions']),
                audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out=Path('artifacts/method-routing-views-v1');out.mkdir(exist_ok=True);body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'views.jsonl').write_text(body)
    report=dict(views=len(rows),experts=len({r['expert_id'] for r in rows}),rejected_or_unresolved=len(rejected),views_sha256=hashlib.sha256(body.encode()).hexdigest(),
        warning='Observed fields passed model citation audit only. Search hints never count as proven cross-domain capability; original evidence is retained for final verification.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
