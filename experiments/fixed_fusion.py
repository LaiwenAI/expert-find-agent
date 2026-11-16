"""Prespecified bounded-score fusion check, before any pruning optimization."""
import hashlib
import json
from pathlib import Path
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.packing_analysis import paired_interval


def run():
    protocol=dict(version=1,alpha=.5,prior='1-r/(n-1) for zero-based Model1 rank within fixed pool; singleton=1',
        slow='Frozen cross-encoder source-max score, uncalibrated bounded ranking feature',
        models=['ms-marco-TinyBERT-L-2-v2','ms-marco-MiniLM-L-12-v2'],datasets=['NIPS','SIGIR'],
        rule='Equal bounded feature weights, fixed before analysis; no label tuning. No claimed pruning speedup.')
    marker=Path('docs/exploration/fixed-fusion-protocol.json')
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed fusion protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    report=dict(protocol=protocol,datasets={})
    for dataset in protocol['datasets']:
        root,_,_,labels=load(dataset)
        table={}
        for model in protocol['models']:
            folder=root/'crossencoder-source-max-v1'/model
            missing=[q for q in labels['soft'] if not (folder/f'{q}.json').exists()]
            if missing:
                table[model]=dict(complete=False,missing=len(missing))
                continue
            rows=[]
            for qid in sorted(labels['soft']):
                ce=json.loads((folder/f'{qid}.json').read_text())
                source=json.loads((root/'packing-shared-v2'/f'{qid}-similarity.json').read_text())
                aliases=source['aliases']
                if ce['aliases_sha256']!=hashlib.sha256(json.dumps(aliases).encode()).hexdigest():raise ValueError('Alias mismatch')
                original=json.loads((root/'language-models-v2'/f'query-{qid}.json').read_text())
                prior=[eid for eid in original['full_rankings']['model1-document'] if eid in set(aliases)]
                if set(prior)!=set(aliases):raise ValueError('Incomplete pool prior')
                base={eid:1-i/(len(prior)-1) if len(prior)>1 else 1. for i,eid in enumerate(prior)}
                slow={aliases[r['id']]:r['score'] for r in ce['scores']}
                if set(slow)!=set(base) or any(not 0<=v<=1 for v in slow.values()):raise ValueError('Invalid slow feature')
                score={eid:.5*base[eid]+.5*slow[eid] for eid in base}
                rank=sorted(score,key=lambda eid:(-score[eid],eid))
                rankings={'model1-in-pool':prior,'crossencoder':ce['ranking'],'fixed-fusion':rank}
                rows.append({task:{method:precision_metrics(ranking,truth[qid]) for method,ranking in rankings.items()} for task,truth in labels.items()})
            metrics={task:{method:{key:sum(r[task][method][key] for r in rows)/len(rows)
                                   for key in ('p@5','p@10','ndcg@10')} for method in rankings} for task in labels}
            differences={task:paired_interval([r[task]['fixed-fusion']['p@5']-r[task]['model1-in-pool']['p@5'] for r in rows]) for task in labels}
            table[model]=dict(complete=True,queries=len(rows),metrics=metrics,fusion_minus_prior_p5=differences)
        report['datasets'][dataset]=table
    report['warning']='Exploratory fixed fusion, not a new statistical theorem or demonstrated new method. SIGIR uses the original fixed-mix top20 pool; Model1-in-pool is distinct from global Model1. Model1 uses full profile statistics, CE uses packed windows. Query bootstrap assumes independent queries, unadjusted for multiple analyses. Optimize execution only if quality warrants it.'
    target=Path('docs/exploration/fixed-fusion-results.json')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
