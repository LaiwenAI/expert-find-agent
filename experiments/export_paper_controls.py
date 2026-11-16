"""Materialize dual-model source-control silver with exact source snapshots."""
import hashlib
import json
from pathlib import Path


def run(audit_path='artifacts/paper-control-blind-audit-v1',control_path='artifacts/paper-backed-controls-v1',output='artifacts/dataset-release/paper-controls-silver-v1'):
    parent=Path(audit_path);summary=json.loads((parent/'summary.json').read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in summary['rows']:
        if item['status']!='dual_model_control_candidate':continue
        record=json.loads((Path(control_path)/(item['id']+'.json')).read_text())
        for view,sources in record['views'].items():
            expected='supported' if view=='full' else 'unknown';audits={}
            for model in ('gpt-4o','gpt-6-luna'):
                p=parent/(item['id']+'-'+view+'-'+model+'.json');a=json.loads(p.read_text())
                if a['status']!='valid' or a['audit']['label']!=expected:raise ValueError('Changed reviewed labels')
                if expected=='supported':
                    citation=a['audit']['citation'];source=sources[citation['evidence_id']]
                    if source['text'][citation['start']:citation['end']]!=citation['text']:raise ValueError('Changed cited source')
                audits[model]=dict(artifact_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),judgment=a['audit'])
            rows.append(dict(id=item['id']+'-'+view,pair_id=item['id'],view=view,condition=record['condition'],label=expected,
                sources=sources,source_sha256={sid:hashlib.sha256(s['text'].encode()).hexdigest() for sid,s in sources.items()},audits=audits))
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(out/'examples.jsonl').write_text(text)
    report=dict(version=1,pairs=len({r['pair_id'] for r in rows}),views=len(rows),examples_sha256=hashlib.sha256(text.encode()).hexdigest(),
        split=f'No train/test split: {len({r["pair_id"] for r in rows})} pairs are an exploratory diagnostic, not a generalization benchmark.',
        license_status='Local only, includes original source snapshots; publication requires source redistribution review.',
        limitations='Dual-model same-family silver. Condition-specific support in an attributed snapshot, not expert inability or full industrial fit. Model agreement is not human gold.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--audit',default='artifacts/paper-control-blind-audit-v1');parser.add_argument('--controls',default='artifacts/paper-backed-controls-v1');parser.add_argument('--output',default='artifacts/dataset-release/paper-controls-silver-v1');args=parser.parse_args();run(args.audit,args.controls,args.output)
