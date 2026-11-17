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
    def evaluate(subset, memory):
        differences=[];routes=0;insertions=0
        for row in subset:
            q=row['query_id'];v=vector(words(q));matches=[]
            for m in memory:
                score=sum(value*vectors[m['query_id']].get(term,0) for term,value in v.items())
                matches.append((score,m['query_id'],m['reviewer_id']))
            score,_,candidate=min(matches,key=lambda x:(-x[0],x[1],x[2]))
            original=row['rankings']['dense'][:5];revised=list(original)
            if score>=.25:
                routes+=1
                if candidate not in revised:
                    revised=[candidate]+revised[:4];insertions+=1
            if len(revised)!=5 or len(set(revised))!=5 or not set(revised)<=set(reviewers):raise ValueError('Invalid revised ranking')
            qrels=labels['soft'][q]
            before=sum(qrels[e]>0 for e in original)/5;after=sum(qrels[e]>0 for e in revised)/5
            differences.append(after-before)
        return {'queries':len(subset),'routed':routes,'inserted':insertions,'mean_delta_soft_p5':statistics.mean(differences),
                'improved':sum(x>1e-12 for x in differences),'worsened':sum(x< -1e-12 for x in differences),'unchanged':sum(abs(x)<=1e-12 for x in differences)}
    controls=[]
    for rep in range(1000):
        shuffled=[m['reviewer_id'] for m in memory]
        random.Random(20261127+rep).shuffle(shuffled)
        randomized=[dict(m,reviewer_id=e) for m,e in zip(memory,shuffled)]
        controls.append({'old':evaluate(old,randomized),'new':evaluate(new,randomized)})
    factual={'old':evaluate(old,memory),'new':evaluate(new,memory)}
    comparison={}
    for split in ('old','new'):
        ds=[r[split]['mean_delta_soft_p5'] for r in controls]
        comparison[split]={'original':factual[split], 'permutation_mean_delta':statistics.mean(ds),
            'permutation_min_delta':min(ds),'permutation_max_delta':max(ds),
            'permutation_at_least_original':sum(v>=factual[split]['mean_delta_soft_p5']-1e-12 for v in ds)}
    output={'permutation_repetitions':len(controls),'comparison':comparison,'queries':len(rows),'old_queries':len(old),'update_queries':len(update),'new_queries':len(new),
            'feedback_records':len(memory),'similarity_threshold':.25,
            'input_sha256':{'runs':sha(source),'summary':sha(summary),'papers':sha(root/'SIGIR_papers_test.json'),
              'labels':sha(root/'SIGIR_queries_test_soft.json'),'script':sha(Path(__file__))},
            'scope':'Post-hoc paired reviewer-identity permutation over fixed query memories; template feedback is never parsed and this is not an ablation of natural-language understanding. Update positives select the original identities; shuffled assignments generally break that relevance. No labels are consumed from evaluation folds to select memories. Repeated permutations are not independent datasets, causal significance tests or evidence of generalization. Post-hoc development-only simulation on previously explored CoF reviewer self-ratings. Update queries generate templated feedback from the lexical ranking and query-level labels; it is not human correction or a verified explanation. Only update documents fit TF-IDF, no old/new labels enter retrieval. Memory copies a prior relevant reviewer without verifying relevance or evidence for the new query; no LLM, attribution check, end-to-end time, genuine feedback, model learning, or policy safety is measured. Arbitrary threshold and one reused, non-chronological split; no held-out or enterprise-fit claim.'}
    out=Path('artifacts/cof-correction-permutation-v1');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
