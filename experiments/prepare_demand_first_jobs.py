"""Materialize every fixed-demand retrieval result, without outcome filtering."""
import hashlib
import json
from pathlib import Path


def run():
    parent=Path('artifacts/demand-first-paper-retrieval-v1');out=Path('artifacts/demand-first-paper-jobs-v1');out.mkdir(exist_ok=True)
    queries=[json.loads(l) for l in (parent/'queries.jsonl').read_text().splitlines()];jobs=[];parents={}
    for query in queries:
        path=parent/'results'/(query['id']+'.json');row=json.loads(path.read_text());parents[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if row['demand']!=query['demand']:raise ValueError('Changed original demand')
        if any(s['expert_id']!=row['expert_id'] or s['kind']!='paper' or len(s['text'])>1200 for s in row['sources'].values()):raise ValueError('Invalid paper view')
        jobs.append(dict(id=row['id']+'-expert-'+row['expert_id'],parent_demand=row['id'],demand=row['demand'],expert_id=row['expert_id'],sources=row['sources'],status='retrieved_not_labeled'))
    data=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs);target=out/'jobs.jsonl'
    if target.exists() and target.read_text()!=data:raise ValueError('Changed demand-first jobs')
    target.write_text(data);(out/'manifest.json').write_text(json.dumps(dict(jobs=len(jobs),demands=len(queries),parents=parents,jobs_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),warning='Retrieval candidate jobs only. No capability labels, teacher-topic prefilter or outcome-based admission.'),indent=2));print('Prepared',len(jobs),'unlabeled jobs')


if __name__=='__main__':run()
