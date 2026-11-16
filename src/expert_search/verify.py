"""Audit saved benchmark and model artifacts against normalized evidence."""
import json
from pathlib import Path
from .corpus import records
from .benchmark import metrics


def verify():
    evidence={r['id']:r for _,r in records('artifacts/corpus/evidence.jsonl')}
    experts={r['id'] for _,r in records('artifacts/corpus/experts.jsonl')}
    root=Path('artifacts/benchmark')
    manifest=json.loads((root/'manifest.json').read_text())
    excluded=set(manifest['excluded_evidence']); counts={'baseline_runs':0,'model_runs':0,'citations':0}
    for _,r in records(root/'runs.jsonl'):
        assert len(r['ranking'])==len(set(r['ranking']))
        assert set(r['ranking'])<=experts
        assert metrics(r['ranking'],r['qrels'])==r['metrics']
        counts['baseline_runs']+=1
    for p in Path('artifacts/agent-benchmark').glob('paper-*.json'):
        row=json.loads(p.read_text())
        counts['model_runs']+=1
        if row['status']!='ok': continue
        assert metrics(row['ranking'],row['qrels'])==row['metrics']
        for result in row['response']['results']:
            eid=result['expert']['id']
            assert result['evidence']
            for leaf in result['evidence']:
                assert leaf['id'] not in excluded
                assert evidence[leaf['id']]==leaf
                assert leaf['expert_id']==eid
                counts['citations']+=1
    return counts
