"""Simulate textual candidate corrections and test cross-query memory retrieval."""
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from expert_search.cof_benchmark import load
from expert_search.fast_retrieval import FastSearch


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
    keys=sorted(groups);rng=random.Random(20261123);rng.shuffle(keys)
    old=[r for key in keys[:20] for r in groups[key]];update=[r for key in keys[20:40] for r in groups[key]];new=[r for key in keys[40:] for r in groups[key]]
    if len(groups)!=73 or (len(old),len(update),len(new))!=(20,20,33):raise ValueError('Changed grouping')
    def words(q):return Counter(re.findall(r'[a-z0-9]+',text[q]))
    update_tokens={r['query_id']:words(r['query_id']) for r in update}
    df=Counter(t for counts in update_tokens.values() for t in counts)
    def vector(counts):
        v={t:(1+math.log(n))*(math.log((len(update)+1)/(df.get(t,0)+1))+1) for t,n in counts.items() if t in df}
        norm=math.sqrt(sum(x*x for x in v.values()))
        return {t:x/norm for t,x in v.items()} if norm else {}
    vectors={q:vector(c) for q,c in update_tokens.items()}
    # Simulated human-style correction: a reviewer judged relevant to this
    # update query appears in lexical top five but is absent from dense top five.
    memory=[]
    for row in update:
        q=row['query_id'];dense=set(row['rankings']['dense'][:5])
        proposed=next((e for e in row['rankings']['lexical'][:5] if e not in dense and labels['soft'][q][e]>0),None)
        if proposed is not None:
            memory.append({'query_id':q,'reviewer_id':proposed,'feedback':'Consider the named researcher: their profile was rated relevant to this previous paper.'})
    if not memory:raise ValueError('No simulated corrections')
    lexical=FastSearch(root/'unique-profile-corpus',root/'unique-profile-corpus/fast.sqlite')
    def evaluate(subset):
        changes={'ungated':[],'gated':[]};routes=0;insertions={'ungated':0,'gated':0};extra=[]
        for row in subset:
            q=row['query_id'];v=vector(words(q));matches=[]
            for m in memory:
                score=sum(value*vectors[m['query_id']].get(term,0) for term,value in v.items())
                matches.append((score,m['query_id'],m['reviewer_id']))
            score,_,candidate=min(matches,key=lambda x:(-x[0],x[1],x[2]))
            original=row['rankings']['dense'][:5];ungated=list(original);gated=list(original)
            if score>=.25:
                routes+=1
                if candidate not in original:
                    ungated=[candidate]+original[:4];insertions['ungated']+=1
                    start=time.perf_counter()
                    scores=lexical.document_scores(text[q])
                    owner_scores={eid:max((scores.get(pid,0.) for pid in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
                    rank=sorted(owner_scores,key=lambda eid:(-owner_scores[eid],eid))
                    extra.append(time.perf_counter()-start)
                    if candidate in rank[:50] and owner_scores[candidate]>0:
                        gated=[candidate]+original[:4];insertions['gated']+=1
            qrels=labels['soft'][q];before=sum(qrels[e]>0 for e in original)/5
            for name,ranking in (('ungated',ungated),('gated',gated)):
                if len(ranking)!=5 or len(set(ranking))!=5 or not set(ranking)<=set(reviewers):raise ValueError('Invalid ranking')
                changes[name].append(sum(qrels[e]>0 for e in ranking)/5-before)
        return {'queries':len(subset),'routed':routes,'insertions':insertions,
                'extra_gate_calls':len(extra),'mean_extra_gate_seconds_per_call':statistics.mean(extra) if extra else None,
                'arms':{name:{'mean_delta_soft_p5':statistics.mean(values),
                    'improved':sum(x>1e-12 for x in values),'worsened':sum(x< -1e-12 for x in values),
                    'unchanged':sum(abs(x)<=1e-12 for x in values)} for name,values in changes.items()}}
    result={'queries':len(rows),'old_queries':len(old),'update_queries':len(update),'new_queries':len(new),
            'feedback_records':len(memory),'similarity_threshold':.25,'lexical_gate_depth':50,
            'old':evaluate(old),'new':evaluate(new),
            'input_sha256':{'runs':sha(source),'summary':sha(summary),'papers':sha(root/'SIGIR_papers_test.json'),
              'reviewers':sha(root/'SIGIR_reviewers_test.json'),'labels':sha(root/'SIGIR_queries_test_soft.json'),
              'lexical_index':sha(root/'unique-profile-corpus/fast.sqlite'),'script':sha(Path(__file__))},
            'scope':'Post-hoc reused public reviewer cohort with simulated feedback and query-similarity memory. A memory candidate is retained only if its maximum lexical paper score is positive and among 50 reviewer scores. Lexical matching is a retrieval heuristic, not verified source entailment, individual contribution or suitability. It adds a serial full-lexical scoring step for each potential insertion; per-call exploratory CPU timing excludes index load, query encoding, memory lookup, source reading, model reasoning, feedback cost and serving. No human corrections, independent labels, generalization or enterprise end-to-end claim.'}
    out=Path('artifacts/cof-memory-evidence-gate-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
