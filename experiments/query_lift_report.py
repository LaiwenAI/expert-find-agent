"""Recompute query-association diagnostics from frozen expert rankings."""
import json
from pathlib import Path
from expert_search.cof_benchmark import load
from expert_search.query_lift import query_lift


def run():
    root,_,_,labels=load('SIGIR');methods={}
    for folder in ('baseline-api','enrichment-v1'):
        for qid in sorted(labels['soft']):
            row=json.loads((root/folder/(qid+'.json')).read_text())
            for method,result in row['results']['soft'].items():methods.setdefault(method,{})[qid]=result['ranking']
    report=dict(dataset='SIGIR',queries=len(labels['soft']),metrics={task:{method:query_lift(rankings,truth,k=5)
        for method,rankings in methods.items()} for task,truth in labels.items()},
        warning='Exploratory supplement to absolute precision, not a validated replacement or established novel metric. Query-independent utility may be legitimate. Query permutation diagnoses association in this fixed label distribution; it does not prove annotation bias or real-world skill.')
    Path('docs/exploration/query-lift-sigir.json').write_text(json.dumps(report,indent=2))
    for task,table in report['metrics'].items():
        for method,result in table.items():print(task,method,'actual',round(result['observed_gain']['lower'],4),'shuffle',round(result['query_shuffle_gain']['lower'],4),'lift',round(result['lower'],4))


if __name__=='__main__':run()
