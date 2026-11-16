"""Build blinded review inputs for constructed source-control cases."""
import hashlib
import json
import random
from pathlib import Path
from expert_search.source_counterfactual import build_views


def build():
    parent=Path('artifacts/source-counterfactual-demand-ranked-local-v4')
    jobs={j['id']:j for j in map(json.loads,(parent/'jobs.jsonl').read_text().splitlines())}
    cases=[]
    for path in sorted((parent/'generation-only').glob('*source-cf*.json')):
        result=json.loads(path.read_text())
        if result['status']=='constructed_pending_independent_audit':
            cases.append((path,jobs[result['id']],result))
    for path in sorted(Path('artifacts/witness-first-controls-v1').glob('*/result.json')):
        result=json.loads(path.read_text())
        if result['status']=='constructed_pending_independent_audit':
            cases.append((path,result['job'],result))
    rows=[]
    private=[]
    for path,job,result in cases:
        views=build_views(job,result['example'])
        if views!=result['views']:
            raise ValueError('Changed source snapshot')
        example=result['example']
        for view in ('full','control'):
            identifier=hashlib.sha256((str(path)+'|'+view).encode()).hexdigest()[:20]
            sources=[]
            aliases={}
            for index,source in enumerate(views[view].values()):
                alias=f'source-{index+1}'
                aliases[alias]=source['id']
                sources.append(dict(id=alias,kind=source['kind'],title=source.get('title'),text=source['text']))
            rows.append(dict(id=identifier,original_demand=job['demand'],
                conditions=[dict(id='a',text=example['topics'][0]),dict(id='b',text=example['topics'][1]),
                            dict(id='relation',text=example['relation'])],sources=sources,
                review=dict(judgments=[dict(id=i,label=None,evidence_id=None,quote=None,reason=None)
                                       for i in ('a','b','relation')],demand_related=None,demand_quote=None,
                            reviewer_id=None)))
            private.append(dict(id=identifier,parent=str(path),parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                view=view,source_aliases=aliases))
    random.Random(20261119).shuffle(rows)
    root=Path('artifacts/blind-source-review-v1');root.mkdir(exist_ok=True)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    marker=root/'inputs.jsonl'
    if marker.exists() and marker.read_text()!=payload:
        raise ValueError('Changed review packet')
    marker.write_text(payload)
    (root/'private-mapping.json').write_text(json.dumps(private,indent=2))
    (root/'README.txt').write_text('Review inputs.jsonl independently. Judge each complete condition from the supplied sources as supported, contradicted, or unknown. Missing evidence means unknown. For supported, quote a unique contiguous passage and its source ID. Check concrete technical relevance to the original demand and quote that requirement. Future project targets are not prerequisites of past achievement. Mark ambiguity in reason. Record reviewer_id. Reviewers should not read private-mapping.json or model outputs before submitting. These forms currently contain no human judgments.\n')
    report=dict(items=len(rows),source_cases=len(cases),human_judgments=0,
                input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                location=str(root),status='Prepared for independent review',
                scope='Local original-source review packet. Contains enterprise and teacher text and is excluded from the source archive.')
    (root/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':build()
