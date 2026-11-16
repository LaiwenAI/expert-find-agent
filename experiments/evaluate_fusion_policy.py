"""Evaluate the frozen eligibility rule; never select per-query using labels."""
import argparse
import hashlib
import json
from pathlib import Path
from expert_search.statistics import paired_interval


def run(datasets):
    path=Path('docs/exploration/selectivity-fusion-protocol-v1.json');policy=json.loads(path.read_text());out=Path('artifacts/selectivity-fusion-policy-v1');out.mkdir(exist_ok=True);reports={}
    for dataset in datasets:
        parent=Path('artifacts')/('cof-source-fusion-'+dataset.lower()+'-v1');summary=json.loads((parent/'summary.json').read_text());rows=[]
        for file in sorted(parent.glob('*.json')):
            if file.name in ('summary.json','protocol.json','diagnostics.json'):continue
            row=json.loads(file.read_text())
            if 'rankings' not in row:continue
            selected='source_top3' if len(row['rankings']['qwen_max'])<=policy['threshold'] else 'qwen_max'
            rows.append(dict(query_id=row['query_id'],selected=selected,metrics=row['metrics'][selected],baselines=row['metrics']))
        if len(rows)!=summary['queries']:raise ValueError('Incomplete frozen-policy dataset')
        tasks=('soft','hard');metrics={t:{m:sum(r['metrics'][t][m] for r in rows)/len(rows) for m in ('p@5','ndcg@10')} for t in tasks}
        contrasts={b:{t:paired_interval([r['metrics'][t]['p@5']-r['baselines'][b][t]['p@5'] for r in rows]) for t in tasks} for b in summary['protocol']['methods']}
        report=dict(dataset=dataset,queries=len(rows),policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_summary_sha256=hashlib.sha256((parent/'summary.json').read_bytes()).hexdigest(),selections={m:sum(r['selected']==m for r in rows) for m in ('qwen_max','source_top3')},metrics=metrics,contrasts=contrasts,rows=rows,warning='Rule chosen on SIGIR/NIPS; extension outputs checked without retuning. Benchmarks have prior baseline exposure, not completely untouched tests. No per-query label-driven selection. Query CIs unadjusted for multiple analyses; no SOTA or industrial-fit claim.')
        (out/(dataset+'.json')).write_text(json.dumps(report,indent=2));reports[dataset]={k:report[k] for k in ('queries','selections','metrics')}
    (out/'summary.json').write_text(json.dumps(dict(policy=policy,datasets=reports),indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['SIGIR','NIPS','KDD','SciRepEval']);run(parser.parse_args().datasets)
