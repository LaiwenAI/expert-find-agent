"""Progressive expert search interface with separate retrieval/explanation clocks.
This research prototype ranks research-area fields only, not full corpus expertise.
"""
import re
import time
from collections import defaultdict
from pathlib import Path
from .area_retrieval import AreaSearch
from .corpus import records


class FastExpertAgent:
    def __init__(self,corpus='artifacts/corpus',model_client=None,full_database=None,reasoning_effort=None):
        start=time.perf_counter();self.index=AreaSearch(Path(corpus)/'evidence.jsonl')
        self.experts={r['id']:r for _,r in records(Path(corpus)/'experts.jsonl')}
        self.profiles={r['expert_id']:r for r in self.index.evidence};self.client=model_client;self.reasoning_effort=reasoning_effort
        self.literal=defaultdict(set)
        for eid,profile in self.profiles.items():
            for term in profile['text'].splitlines():self.literal[term.strip()].add(eid)
        self.full=None
        if full_database is not None:
            from .fast_retrieval import FastSearch
            self.full=FastSearch(corpus,full_database)
        self.initialization_seconds=time.perf_counter()-start

    def events(self,query,k=5,explain=False):
        if not query.strip() or k<1:raise ValueError('Invalid query or k')
        start=time.perf_counter();explicit=re.findall('「([^」]+)」',query)
        if explicit:
            matches=set(self.literal.get(explicit[0],set()))
            for term in explicit[1:]:matches.intersection_update(self.literal.get(term,set()))
            hits=[dict(id=eid,score=1.,evidence_id=self.profiles[eid]['id'],source=self.profiles[eid]['source']) for eid in sorted(matches)[:k]]
            semantics='Exact explicitly quoted research-area conjunction; not verified actual capability'
        else:
            hits=self.index.search(query,k)['results'];semantics='Research-area BM25 ranking; relevance and capabilities not verified'
        results=[dict(expert=self.experts[h['id']],score=h['score'],evidence=dict(id=h['evidence_id'],text=self.profiles[h['id']]['text'],source=h['source'])) for h in hits]
        first=time.perf_counter()-start
        yield dict(stage='retrieval',results=results,ranking_semantics=semantics,first_result_seconds=first,
            initialization_seconds=self.initialization_seconds,cold_initialization_plus_query_seconds=self.initialization_seconds+first,
            warning='Area-only scope; missing evidence does not imply absent ability; explanation completion measured separately')
        if self.full is not None and not explicit:
            full_hits=self.full.search(query,k,'bm25-vote')
            results=[dict(expert=self.experts[h['id']],score=h['score'],evidence={key:value for key,value in h['evidence'][0].items() if key!='score'}) for h in full_hits]
            yield dict(stage='full_evidence',results=results,total_seconds=time.perf_counter()-start,
                scope='All indexed profile and paper evidence, BM25-vote',warning='New ranking may replace initial field-only ranking; not semantic verification')
        if explain and results:
            from .agent import ExpertAgent
            if self.client is None:
                from laiwenai import LaiwenAI
                self.client=LaiwenAI(model='gpt-6-luna')
            # Reuse ask without building another retrieval corpus.
            helper=ExpertAgent.__new__(ExpertAgent);helper.client=self.client;helper.trace=[];helper.usage={'prompt_tokens':0,'completion_tokens':0}
            if self.reasoning_effort is not None:helper.request_options={'reasoning_effort':self.reasoning_effort}
            answer=helper.ask('explain','Explain evidence-based relevance and missing requirements for these experts. Use only supplied evidence; do not invent capabilities or change ranking. Return {"experts":[{"id":str,"reason":str,"missing":[str],"citation_id":str}]}.',{'query':query,'results':results})
            allowed={r['expert']['id']:r['evidence']['id'] for r in results};valid=[]
            for row in answer.get('experts',[]):
                if isinstance(row,dict) and row.get('id') in allowed and row.get('citation_id')==allowed[row['id']]:valid.append(row)
            yield dict(stage='explanation',experts=valid,total_seconds=time.perf_counter()-start,usage=helper.usage,
                warning='Model explanation; citation ownership validated, semantic correctness not guaranteed')
