"""Shared-snapshot gain contrasts for the frozen focused-output experiment."""
import hashlib
import json
from pathlib import Path
from expert_search.evaluation_gain import paired_actionable_gain_interval


def run():
    root=Path('artifacts/focused-partial-comparison-v1');protocol=json.loads((root/'protocol.json').read_text());jobs={r['id']:r for r in map(json.loads,Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl').read_text().splitlines())};grades={}
    for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):grades.setdefault(r['query_id'],{})[r['expert_id']]=r['grade']
    rows=[];inputs={}
    for qid in protocol['query_ids']:
        path=root/(qid+'.json');value=json.loads(path.read_text());inputs[qid]=hashlib.sha256(path.read_bytes()).hexdigest();rankings={}
        for arm in ['full','focused']:
            review=value['arms'][arm]['review'];ranking=review['result']['ranking'] if review['status']=='structurally_valid' else []
            if any(p['expert_id'] not in jobs[qid]['candidates'] for p in ranking):raise ValueError('Foreign candidate')
            # Both arms receive the same frozen raw source dictionaries, verified
            # by the comparison protocol and original builder; no view expansion.
            rankings[arm]=[p['expert_id'] for p in ranking][:3]
        contrast=paired_actionable_gain_interval(rankings['focused'],rankings['full'],grades[qid],3)
        rows.append(dict(query_id=qid,first='focused',second='full',contrast=contrast))
    n=len(rows);report=dict(queries=n,mean_shared_lower=sum(r['contrast']['lower'] for r in rows)/n,mean_shared_upper=sum(r['contrast']['upper'] for r in rows)/n,mean_independent_lower=sum(r['contrast']['independent_bounds'][0] for r in rows)/n,mean_independent_upper=sum(r['contrast']['independent_bounds'][1] for r in rows)/n,cancelled_shared_unjudged=sum(r['contrast']['cancelled_shared_unjudged'] for r in rows),inputs_sha256=inputs,code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/expert_search/evaluation_gain.py')]},scope='Diagnostic on reused development cohort. Known labels are same-model silver; shared-unknown cancellation sharpens identification only, not independent accuracy or statistical confidence. Failed arms give no completed ranking. Source views unchanged across arms.')
    out=Path('artifacts/paired-view-gain-v1');out.mkdir(exist_ok=True);(out/'contrasts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
