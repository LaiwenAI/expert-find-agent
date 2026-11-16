"""Paired scale-check comparison with the original lexical candidate protocol."""
import json
from pathlib import Path
from urllib.parse import quote
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.packing_analysis import paired_interval


def compare(dataset):
    root,_,_,labels=load(dataset)
    qids=sorted(labels['soft'])
    missing=[q for q in qids if not (root/'baseline-lexical'/f"{quote(q,safe='')}.json").exists()]
    if missing:
        return dict(dataset=dataset,complete=False,completed=len(qids)-len(missing),planned=len(qids))
    copied=set(json.loads(Path(f'docs/exploration/cof-{dataset.lower()}-audit.json').read_text())['exact_query_texts_in_profiles'])
    rows=[]
    for qid in qids:
        filename=f"{quote(qid,safe='')}.json"
        lm=json.loads((root/'scale-check-v1'/filename).read_text())
        lexical=json.loads((root/'baseline-lexical'/filename).read_text())
        if lm['query_id']!=qid or lexical['query_id']!=qid:
            raise ValueError('Wrong query ID')
        for task,truth in labels.items():
            if lm['metrics'][task]!=precision_metrics(lm['ranking'],truth[qid]):
                raise ValueError('Changed LM metrics')
            for method in ('bm25-max','bm25-top3'):
                record=lexical['results'][task][method]
                if record['metrics']!=precision_metrics(record['ranking'],truth[qid]):
                    raise ValueError('Changed lexical metrics')
        rows.append((qid,lm,lexical))
    cohorts={'original':rows,'exclude_exact_copy_queries':[r for r in rows if r[0] not in copied]}
    return dict(dataset=dataset,complete=True,
        cohorts={name:dict(queries=len(selected),lm_minus_lexical={method:{task:{metric:paired_interval([
            lm['metrics'][task][metric]-lexical['results'][task][method]['metrics'][metric]
            for _,lm,lexical in selected]) for metric in ('p@5','ndcg@10')} for task in labels}
            for method in ('bm25-max','bm25-top3')}) for name,selected in cohorts.items()},
        scope='Exploratory query bootstrap on reused released labels. Absolute quality and timing remain in the source reports. Exact-copy exclusion does not cover near duplicates.')


if __name__=='__main__':
    for dataset in ('KDD','SciRepEval'):
        report=compare(dataset)
        Path(f'docs/exploration/cof-scale-{dataset.lower()}-comparison.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='cohorts'},indent=2))
