"""Entry point for research experiments; existing search CLI stays compatible."""
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['public-lexical','public-hybrid','cmu-lexical','cmu-dense','comlq-lexical','cof-sigir-lexical','cof-sigir-dense','synthetic-diagnostics','synthetic-select'])
    a=p.parse_args()
    if a.task=='public-lexical':
        from .public_benchmark import run
        result=run()
    elif a.task=='public-hybrid':
        from .hybrid_benchmark import run
        result=run()
    elif a.task=='cmu-lexical':
        from .cmu_baseline import run
        result=run()
    elif a.task=='cmu-dense':
        from .cmu_dense_benchmark import run
        result=run()
    elif a.task=='comlq-lexical':
        from .comlq import prepare
        from .public_benchmark import run
        prepare()
        result=run(corpus='artifacts/public/comlq/adapted',output='artifacts/public/comlq/results',dataset_name='ComLQ')
    elif a.task in ('cof-sigir-lexical','cof-sigir-dense'):
        from .cof_benchmark import run,encode_papers
        dense=a.task=='cof-sigir-dense'
        if dense:encode_papers('SIGIR')
        result=run(dataset='SIGIR',use_dense=dense)
    elif a.task=='synthetic-diagnostics':
        from .synthetic_diagnostics import run
        result=run()
    else:
        from .filter_synthetic import select
        result=select('artifacts/synthesis-evidence-first','artifacts/synthesis-evidence-first-audit','artifacts/synthetic-evidence-first-selected')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
