"""Simulate textual candidate corrections and test cross-query memory retrieval."""
import hashlib
import json
import math
import random
import re
import statistics
import time
import numpy as np
from collections import Counter
from pathlib import Path
from expert_search.cof_benchmark import load
from expert_search.fast_retrieval import FastSearch
from expert_search.resident_candidate_bm25 import ResidentCandidateBM25
from expert_search.retrieval import tokens


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
    resident=ResidentCandidateBM25(root/'unique-profile-corpus/fast.sqlite',adaptive_intersection=True)
    # Coauthored papers can be associated with multiple reviewers; each
    # posting is expanded to every owner before the per-term maxima are made.
    start=time.perf_counter()
    experts=sorted(reviewers);positions={e:i for i,e in enumerate(experts)}
    owners={pid:[] for pid in resident.ids}
    for eid,r in reviewers.items():
        for pid in set(r['papers']):owners[pid].append(positions[eid])
    term_bounds={}
    for term,(lo,hi) in resident.offsets.items():
        maxima={}
        for pos,value in zip(resident.indices[lo:hi],resident.values[lo:hi]):
            for owner in owners[resident.ids[int(pos)]]:
                maxima[owner]=max(maxima.get(owner,0.),float(value))
        term_bounds[term]=maxima
    bound_init=time.perf_counter()-start
    def certified(query,candidate):
        candidate_ids=sorted(set(reviewers[candidate]['papers']))
        candidate_scores=resident.selected_scores(query,candidate_ids)
        threshold=max(candidate_scores.values(),default=0.)
        if threshold<=0:return False,0
        upper=np.zeros(len(experts),dtype=np.float64)
        for term in sorted(set(tokens(query))):
            for owner,value in term_bounds.get(term,{}).items():
                upper[owner]=np.nextafter(upper[owner]+value,np.inf)
        possible=[e for e in experts if e!=candidate and upper[positions[e]]>=threshold]
        sources=sorted({pid for e in possible for pid in reviewers[e]['papers']})
        exact=resident.selected_scores(query,sources)
        ahead=0
        for eid in possible:
            value=max((exact[pid] for pid in set(reviewers[eid]['papers'])),default=0.)
            if value>threshold or (value==threshold and eid<candidate):ahead+=1
        return ahead<50,len(possible)
    def evaluate(subset):
        changes={'ungated':[],'gated':[],'local':[]};routes=0;insertions={'ungated':0,'gated':0,'local':0};extra=[];local_times=[];exact_checks=0;certified_times=[];possible_counts=[];certified_checks=0
        for row in subset:
            q=row['query_id'];v=vector(words(q));matches=[]
            for m in memory:
                score=sum(value*vectors[m['query_id']].get(term,0) for term,value in v.items())
                matches.append((score,m['query_id'],m['reviewer_id']))
            score,_,candidate=min(matches,key=lambda x:(-x[0],x[1],x[2]))
            original=row['rankings']['dense'][:5];ungated=list(original);gated=list(original);local=list(original)
            if score>=.25:
                routes+=1
                if candidate not in original:
                    ungated=[candidate]+original[:4];insertions['ungated']+=1
                    start=time.perf_counter()
                    scores=lexical.document_scores(text[q])
                    owner_scores={eid:max((scores.get(pid,0.) for pid in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
                    rank=sorted(owner_scores,key=lambda eid:(-owner_scores[eid],eid))
                    extra.append(time.perf_counter()-start)
                    candidate_sources=sorted(set(reviewers[candidate]['papers']))
                    start=time.perf_counter()
                    local_scores=resident.selected_scores(text[q],candidate_sources)
                    local_times.append(time.perf_counter()-start)
                    if any(not math.isclose(local_scores[pid],scores.get(pid,0.),rel_tol=1e-12,abs_tol=1e-12) for pid in candidate_sources):raise ValueError('Candidate scores differ')
                    exact_checks+=1
                    start=time.perf_counter()
                    admitted,possible=certified(text[q],candidate)
                    certified_times.append(time.perf_counter()-start);possible_counts.append(possible)
                    if admitted != (candidate in rank[:50] and owner_scores[candidate]>0):raise ValueError('Certified gate differs from global')
                    certified_checks+=1
                    if max(local_scores.values(),default=0)>0:
                        local=[candidate]+original[:4];insertions['local']+=1
                    if candidate in rank[:50] and owner_scores[candidate]>0:
                        gated=[candidate]+original[:4];insertions['gated']+=1
            qrels=labels['soft'][q];before=sum(qrels[e]>0 for e in original)/5
            for name,ranking in (('ungated',ungated),('gated',gated),('local',local)):
                if len(ranking)!=5 or len(set(ranking))!=5 or not set(ranking)<=set(reviewers):raise ValueError('Invalid ranking')
                changes[name].append(sum(qrels[e]>0 for e in ranking)/5-before)
        return {'queries':len(subset),'routed':routes,'insertions':insertions,
                'extra_gate_calls':len(extra),'mean_extra_gate_seconds_per_call':statistics.mean(extra) if extra else None,
                'mean_local_seconds_per_call':statistics.mean(local_times) if local_times else None,'candidate_score_checks':exact_checks,'certified_gate_checks':certified_checks,
                'mean_certified_seconds':statistics.mean(certified_times) if certified_times else None,
                'mean_possible_competitors':statistics.mean(possible_counts) if possible_counts else None,
                'arms':{name:{'mean_delta_soft_p5':statistics.mean(values),
                    'improved':sum(x>1e-12 for x in values),'worsened':sum(x< -1e-12 for x in values),
                    'unchanged':sum(abs(x)<=1e-12 for x in values)} for name,values in changes.items()}}
    result={'queries':len(rows),'old_queries':len(old),'update_queries':len(update),'new_queries':len(new),
            'feedback_records':len(memory),'similarity_threshold':.25,'lexical_gate_depth':50,'resident_initialization_seconds':resident.initialization_seconds,'bound_initialization_seconds':bound_init,
            'old':evaluate(old),'new':evaluate(new),
            'input_sha256':{'runs':sha(source),'summary':sha(summary),'papers':sha(root/'SIGIR_papers_test.json'),
              'reviewers':sha(root/'SIGIR_reviewers_test.json'),'labels':sha(root/'SIGIR_queries_test_soft.json'),
              'lexical_index':sha(root/'unique-profile-corpus/fast.sqlite'),'resident_code':sha(Path('src/expert_search/resident_candidate_bm25.py')),'lexical_code':sha(Path('src/expert_search/fast_retrieval.py')),'script':sha(Path(__file__))},
            'scope':'Candidate threshold certification uses upward-rounded per-term owner maxima and exact scores for all owners whose bound reaches the candidate score, including lexicographic ties. It was checked against global membership on this sample only, not semantic relevance. Timings include query-specific bounds and competitor scoring but exclude bound construction. Local gate checks only positive BM25 among the proposed reviewer sources and is not equivalent to the global top-50 gate. Exact candidate scores are checked against full lexical scores, but no output equivalence is claimed. Timings are one sequential observation per case, not independent or end-to-end. Post-hoc reused public reviewer cohort with simulated feedback and query-similarity memory. The global gate requires a positive maximum score and top-50 position; the local gate requires only a positive candidate-source score. Lexical matching is a retrieval heuristic, not verified source entailment, individual contribution or suitability. It adds a serial full-lexical scoring step for each potential insertion; per-call exploratory CPU timing excludes index load, query encoding, memory lookup, source reading, model reasoning, feedback cost and serving. No human corrections, independent labels, generalization or enterprise end-to-end claim.'}
    out=Path('artifacts/cof-memory-certified-gate-v2');out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
