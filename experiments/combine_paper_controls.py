"""Combine chronological silver cohorts with original demand context and overlap audit."""
import argparse
import hashlib
import json
from pathlib import Path
from expert_search.source_counterfactual import normalized


def run(include_demand_first=False):
    cohorts=[('development','artifacts/dataset-release/paper-controls-silver-v1','artifacts/paper-backed-overlap-v1/jobs.jsonl','artifacts/paper-backed-generation-v1'),
        ('new_cohort','artifacts/dataset-release/expanded-paper-controls-silver-v1','artifacts/expanded-paper-backed-v1/jobs.jsonl','artifacts/expanded-paper-generation-v1')]
    if include_demand_first:
        cohorts.append(('demand_first','artifacts/dataset-release/demand-first-paper-controls-silver-v1','artifacts/demand-first-paper-jobs-v1/jobs.jsonl','artifacts/demand-first-paper-generation-v1'))
    rows=[];resources={};manifests={}
    for name,directory,jobs_path,generation in cohorts:
        parent=Path(directory);body=(parent/'examples.jsonl').read_bytes();manifest=json.loads((parent/'manifest.json').read_text())
        if hashlib.sha256(body).hexdigest()!=manifest['examples_sha256']:raise ValueError('Changed admitted dataset')
        manifests[name]=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest();jobs={j['id']:j for j in map(json.loads,Path(jobs_path).read_text().splitlines())}
        resources[name]=dict(experts=set(),demands=set(),condition_text=set(),source_text=set())
        for row in map(json.loads,body.decode().splitlines()):
            job=jobs[row['pair_id']];proposal=json.loads((Path(generation)/(row['pair_id']+'.json')).read_text())['proposal']
            for sid,source in row['sources'].items():
                if source['expert_id']!=job['expert_id'] or hashlib.sha256(source['text'].encode()).hexdigest()!=row['source_sha256'][sid]:raise ValueError('Changed source owner/text')
                resources[name]['source_text'].add(hashlib.sha256(normalized(source['text']).encode()).hexdigest())
            resources[name]['experts'].add(job['expert_id']);resources[name]['demands'].add(job['parent_demand']);resources[name]['condition_text'].add(normalized(row['condition']))
            rows.append(dict(row,cohort=name,original_demand=job['demand'],parent_demand=job['parent_demand'],expert_id=job['expert_id'],remaining_demand_gaps=proposal['remaining_demand_gaps']))
    overlap={key:len(resources['development'][key]&resources['new_cohort'][key]) for key in resources['development']}
    out=Path('artifacts/dataset-release/paper-controls-combined-v3' if include_demand_first else 'artifacts/dataset-release/paper-controls-combined-v2');out.mkdir(exist_ok=True);body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    (out/'examples.jsonl').write_text(body)
    report=dict(version=3 if include_demand_first else 2,pairs=len({r['pair_id'] for r in rows}),views=len(rows),cohorts={name:len({r['pair_id'] for r in rows if r['cohort']==name}) for name,*_ in cohorts},
        normalized_exact_cross_cohort_overlap=overlap,parent_manifests=manifests,examples_sha256=hashlib.sha256(body.encode()).hexdigest(),
        warning='Chronological exploratory cohorts, not human gold. Near duplicate semantic overlap untested. Labels describe source-snapshot support for narrow conditions, not inability or complete enterprise-fit. Local source snapshots are excluded from source-code archive.')
    if include_demand_first:
        names=list(resources);report['pairwise_normalized_exact_overlap']={a+'__'+b:{key:len(resources[a][key]&resources[b][key]) for key in resources[a]} for i,a in enumerate(names) for b in names[i+1:]}
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-demand-first',action='store_true');run(parser.parse_args().include_demand_first)
