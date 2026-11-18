"""Simulate textual candidate corrections and test cross-query memory retrieval."""
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from expert_search.cof_benchmark import load


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root,papers,reviewers,labels=load('SIGIR');source=Path('artifacts/cof-single-query-v1/rows.json')
    summary=source.parent/'summary.json'
    if sha(source)!=json.loads(summary.read_text())['rows_sha256']:raise ValueError('Changed runs')
    rows=json.loads(source.read_text())
    if len(rows)!=73 or {r['query_id'] for r in rows}!=set(labels['soft']):raise ValueError('Changed cohort')
    text={r['query_id']:(papers[r['query_id']].get('title','')+' '+papers[r['query_id']].get('abstract','')).casefold() for r in rows}
    groups={}
    for row in rows:
        q=row['query_id'];key=hashlib.sha256(' '.join(text[q].split()).encode()).hexdigest()
        groups.setdefault(key,[]).append(row)
    if len(groups)!=73:raise ValueError('Changed grouping')
    def words(q):return Counter(re.findall(r'[a-z0-9]+',text[q]))
    def one(seed):
        keys=sorted(groups);rng=random.Random(seed);rng.shuffle(keys)
        old=[r for key in keys[:20] for r in groups[key]]
        update=[r for key in keys[20:40] for r in groups[key]]
        new=[r for key in keys[40:] for r in groups[key]]
        update_tokens={r['query_id']:words(r['query_id']) for r in update}
        df=Counter(t for counts in update_tokens.values() for t in counts)
        def vector(counts):
            v={t:(1+math.log(n))*(math.log((len(update)+1)/(df.get(t,0)+1))+1) for t,n in counts.items() if t in df}
            norm=math.sqrt(sum(x*x for x in v.values()))
            return {t:x/norm for t,x in v.items()} if norm else {}
        vectors={q:vector(c) for q,c in update_tokens.items()}
        memory=[]
        for row in update:
            q=row['query_id'];dense=set(row['rankings']['dense'][:5])
            proposed=next((e for e in row['rankings']['lexical'][:5] if e not in dense and labels['soft'][q][e]>0),None)
            if proposed is not None:memory.append((q,proposed))
        def evaluate(subset):
            differences=[];insertions=0
            for row in subset:
                q=row['query_id'];v=vector(words(q));original=row['rankings']['dense'][:5];revised=list(original)
                matches=[(sum(value*vectors[mq].get(term,0) for term,value in v.items()),mq,eid) for mq,eid in memory]
                if matches:
                    score,_,candidate=min(matches,key=lambda x:(-x[0],x[1],x[2]))
                    if score>=.25 and candidate not in revised:revised=[candidate]+revised[:4];insertions+=1
                truth=labels['soft'][q]
                differences.append((sum(truth[e]>0 for e in revised)-sum(truth[e]>0 for e in original))/5)
            return {'queries':len(subset),'insertions':insertions,'mean_delta':statistics.mean(differences),
                    'improved':sum(x>0 for x in differences),'worsened':sum(x<0 for x in differences)}
        return {'seed':seed,'feedback_records':len(memory),'old':evaluate(old),'new':evaluate(new)}
    trials=[one(20262000+i) for i in range(30)]
    output={'splits':len(trials),'queries':len(rows),'trials':trials,
            'summary':{cohort:{'mean_delta':statistics.mean(t[cohort]['mean_delta'] for t in trials),
              'splits_positive':sum(t[cohort]['mean_delta']>0 for t in trials),
              'splits_negative':sum(t[cohort]['mean_delta']<0 for t in trials),
              'mean_insertions':statistics.mean(t[cohort]['insertions'] for t in trials)} for cohort in ('old','new')},
            'input_sha256':{'runs':sha(source),'summary':sha(summary),'papers':sha(root/'SIGIR_papers_test.json'),
              'labels':sha(root/'SIGIR_queries_test_soft.json'),'script':sha(Path(__file__))},
            'scope':'Thirty overlapping, random post-hoc splits of the same 73 previously explored CoF queries; split results are dependent, not independent replications or confidence intervals. Simulated corrections use update-query reviewer self-ratings and lexical candidates; no natural-language parsing, actual feedback, evidence labels, temporal transfer, end-to-end cost or training. Similarity threshold and method were chosen from earlier exploration. TF-IDF vocabulary is fit on each update fold only. Old and new folds have identical sampling roles here; neither is a longitudinal retention test. Reviewer self-ratings are not enterprise labels.'}
    out=Path('artifacts/cof-correction-memory-splits-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
