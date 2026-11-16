"""Report within/between-mode agreement without assuming model labels are gold."""
import json
from pathlib import Path


def labels(row):
    return {(eid,rid):a['level'] for eid,o in row['outcomes'].items() if o['status']=='structurally_valid'
            for rid,a in o['result']['assessments'].items()}


def run():
    root=Path('artifacts/compact-assessment-comparison-v1')
    rows=[json.loads(p.read_text()) for p in root.glob('demand-*.json')]
    summary=json.loads((root/'summary.json').read_text());report=dict(within_mode={},label_counts={})
    for mode in ('singleton','batch'):
        equal=total=0;counts={}
        for key in sorted({r['id'] for r in rows}):
            a,b=[labels(next(r for r in rows if r['id']==key and r['mode']==mode and r['repeat']==i)) for i in (0,1)]
            keys=a.keys()&b.keys();equal+=sum(a[k]==b[k] for k in keys);total+=len(keys)
        for row in rows:
            if row['mode']==mode:
                for label in labels(row).values():counts[label]=counts.get(label,0)+1
        report['within_mode'][mode]=dict(agree=equal,comparable=total);report['label_counts'][mode]=counts
    report['cross_mode']=dict(agree=sum(p['agreeing_labels'] for p in summary['paired']),comparable=sum(p['comparable_labels'] for p in summary['paired']))
    report['warning']='Exploratory repeated model labels, not correctness, calibrated uncertainty or equivalence. No applicability gold.'
    (root/'stability.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
