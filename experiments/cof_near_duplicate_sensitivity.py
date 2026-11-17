"""Post-hoc public reviewer benchmark sensitivity to near-duplicate titles."""
import argparse
from collections import Counter
import hashlib
import json
import re
import statistics
from pathlib import Path
from urllib.parse import quote

from expert_search.cof_benchmark import load,precision_metrics
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def title_tokens(text):
    return set(re.findall(r'[a-z0-9]+',text.lower()))


def near_title(a,b):
    """Fixed conservative token Jaccard; not semantic-duplicate detection."""
    x,y=title_tokens(a),title_tokens(b)
    return len(x)>=5 and len(y)>=5 and len(x&y)/len(x|y)>=0.8


def analyze(dataset):
    root,papers,reviewers,labels=load(dataset);folder=Path('artifacts')/('cof-source-fusion-'+dataset.lower()+'-v1')
    summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    methods=protocol['methods'];ids=sorted(labels['soft'])
    if set(ids)!=set(labels['hard']):raise ValueError('Changed query inventory')
    rows=[];record_hashes={}
    for qid in ids:
        if set(labels['soft'][qid])!=set(labels['hard'][qid]):raise ValueError('Changed candidate inventory')
        candidates=set(labels['soft'][qid]);profile_papers={p for e in candidates for p in reviewers[e]['papers']}
        if profile_papers-set(papers):raise ValueError('Unknown profile paper')
        query=papers[qid];title=query.get('title','');normalized=lambda p:re.sub(r'\s+',' ',(p.get('title','')+'\n'+p.get('abstract','')).strip()).casefold()
        qtext=normalized(query);same_id=qid in profile_papers
        exact=bool(qtext) and any(normalized(papers[pid])==qtext for pid in profile_papers)
        near=any(near_title(title,papers[pid].get('title','')) for pid in profile_papers)
        path=folder/(quote(qid,safe='')+'.json');record_hashes[path.name]=sha(path);row=json.loads(path.read_text())
        if row['query_id']!=qid:raise ValueError('Wrong ranking record')
        for method in methods:
            ranking=row['rankings'][method]
            if len(set(ranking))!=len(ranking) or set(ranking)!=candidates:raise ValueError('Changed candidate ranking')
            for task in labels:
                computed=precision_metrics(ranking,labels[task][qid]);stored=row['metrics'][method][task]
                if any(abs(computed[m]-stored[m])>1e-10 for m in ('p@5','ndcg@10')):raise ValueError('Changed saved metrics')
        rows.append({'query_id':qid,'same_id':same_id,'exact_text':exact,'near_title':near,'metrics':row['metrics']})
    if len(rows)!=summary['queries']:raise ValueError('Incomplete source results')
    sensitivity={}
    for name,subset in [('all',rows),('exclude_exact_or_near',[r for r in rows if not (r['same_id'] or r['exact_text'] or r['near_title'])])]:
        if not subset:sensitivity[name]={'queries':0};continue
        metrics={task:{method:{m:statistics.mean(r['metrics'][method][task][m] for r in subset) for m in ('p@5','ndcg@10')} for method in methods} for task in labels}
        diffs={task:paired_interval([r['metrics']['source_top3'][task]['p@5']-r['metrics']['qwen_max'][task]['p@5'] for r in subset]) for task in labels}
        sensitivity[name]={'queries':len(subset),'metrics':metrics,'source_top3_minus_qwen_p_at_5':diffs}
    return {'dataset':dataset,'queries':len(rows),'flags':dict(Counter({'same_id':sum(r['same_id'] for r in rows),'exact_text':sum(r['exact_text'] for r in rows),
             'near_title':sum(r['near_title'] for r in rows),'union':sum(r['same_id'] or r['exact_text'] or r['near_title'] for r in rows)})),
            'flagged_query_ids':[r['query_id'] for r in rows if r['same_id'] or r['exact_text'] or r['near_title']],
            'sensitivity':sensitivity,'fusion_protocol_sha256':sha(folder/'protocol.json'),'fusion_summary_sha256':sha(folder/'summary.json'),
            'ranking_record_sha256':record_hashes,'source_sha256':{n:sha(root/n) for n in (f'{dataset}_papers_test.json',f'{dataset}_reviewers_test.json',f'{dataset}_queries_test_soft.json',f'{dataset}_queries_test_hard.json')},
            'scope':'Post-hoc fixed title-token Jaccard >=0.8 with at least five distinct tokens per title, plus same source ID or exact normalized title/abstract. Whole flagged queries are excluded, not re-ranked. Titles may refer to legitimate related work; threshold misses semantic duplicates and pretraining contamination. Reused public reviewer labels are not industrial expert fit. Bootstrap assumes independent queries and is unadjusted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for dataset in ('SIGIR','NIPS','KDD','SciRepEval'):
        report=analyze(dataset);(a.output/(dataset+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(dataset,json.dumps({'queries':report['queries'],'flags':report['flags'],'retained':report['sensitivity']['exclude_exact_or_near']['queries']}),flush=True)
