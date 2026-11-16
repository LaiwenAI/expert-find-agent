"""Measure raw versus dependency-aware claim coverage in actual model outputs."""
import json
from pathlib import Path
from expert_search.constraint_metrics import evidence_gain
from expert_search.scope_metrics import scoped_gain


def run():
    root=Path('artifacts/witness-control-auxiliary-audit-v1')
    requirements={
        'a':dict(weight=1,mandatory=False,kind='atomic'),
        'b':dict(weight=1,mandatory=False,kind='atomic'),
        'relation':dict(weight=1,mandatory=False,kind='relation',dependencies=['a','b'])}
    rows=[]
    unresolved=[]
    for path in sorted(root.glob('*.json')):
        if path.name in ('protocol.json','summary.json'):
            continue
        record=json.loads(path.read_text())
        if record['status']!='valid':
            unresolved.append(path.name)
            continue
        labels=record['labels']
        raw=evidence_gain(requirements,labels)
        scoped=scoped_gain(requirements,labels)
        rows.append(dict(record=path.name,raw_coverage=raw['supported_coverage'],
            dependency_coverage=scoped['supported_coverage'],
            blocked_relation=labels['relation']=='supported' and scoped['resolved_judgments']['relation']!='supported'))
    report=dict(schema_valid_views=len(rows),unresolved_views=unresolved,
        dependency_inconsistent_views=sum(r['blocked_relation'] for r in rows),rows=rows,
        interpretation='Observed model-output consistency diagnostic. Both coverage values depend on model judgments. Human relevance labels and metric validity remain unverified.')
    Path('docs/exploration/auxiliary-scope-analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
