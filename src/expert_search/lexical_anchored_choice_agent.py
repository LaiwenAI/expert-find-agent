"""Training-free lexical span proposals followed by one native choice request."""
from collections import Counter
import hashlib
import math
import time

from .candidate_evidence_pipeline import CandidateEvidencePipeline
from .choice_judgments import decide
from .progressive_decision_agent import FIT_CHOICES,FIT_INSTRUCTION
from .retrieval import tokens

STOP={'the','and','for','with','that','this','from','are','was','were','have','has','had','into','their','which','using','use','our','its','can','also','based','between','over','under','such','these','those','through','than','not','but','all','each','paper','study','approach','method','methods','model','models','data','results','result'}


def term_set(text):
    return {t for t in tokens(text) if t not in STOP and (len(t)>=3 or any('\u4e00'<=c<='\u9fff' for c in t))}


def propose_pair(query,sources,width=400,source_chars=1200):
    """Return the highest lexical-overlap pair, or None when overlap is absent."""
    if not isinstance(query,str) or not query.strip() or type(width)!=int or width<1 or type(source_chars)!=int or source_chars<1:raise ValueError('Invalid span budget')
    demands=[{'start':i,'end':min(i+width,len(query)),'quote':query[i:i+width]} for i in range(0,len(query),width) if query[i:i+width].strip()]
    if len(demands)>32:raise ValueError('Demand exceeds span budget')
    evidence=[]
    for source in sources:
        text=source['text'][:source_chars];fingerprint=hashlib.sha256(text.encode()).hexdigest()
        for pos in range(0,len(text),width):
            piece=text[pos:pos+width]
            if piece.strip():evidence.append({'evidence_id':source['id'],'expert_id':source['expert_id'],'start':pos,'end':pos+len(piece),'quote':piece,'source_view_sha256':fingerprint})
    if not evidence:return None
    qsets=[term_set(d['quote']) for d in demands];esets=[term_set(e['quote']) for e in evidence]
    frequencies=Counter(t for group in qsets+esets for t in group);n=len(qsets)+len(esets);best=None
    for i,q in enumerate(qsets):
        for j,e in enumerate(esets):
            shared=q&e
            if not shared:continue
            weight=sum(1+math.log((n+1)/(frequencies[t]+1)) for t in sorted(shared))
            key=(weight,-i,-j)
            if best is None or key>best[0]:best=(key,i,j,len(shared))
    if best is None:return None
    return {'demand_anchor':demands[best[1]],'source_anchor':evidence[best[2]],'shared_terms':best[3],
            'scope':'Deterministic lexical overlap is a span proposal only, not semantic support, a method bridge or personal credit.'}


class LexicalAnchoredChoiceAgent:
    """Judge proposed span pairs in one request; zero-overlap cases stay unknown."""
    def __init__(self,pipeline=None,client=None,decision_fn=None):
        self.pipeline=pipeline if pipeline is not None else CandidateEvidencePipeline()
        self.client=client;self.decision_fn=decision_fn if decision_fn is not None else decide

    def run(self,query,k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200,span_width=400):
        if not isinstance(query,str) or not query.strip() or any(type(v)!=int for v in (k,candidates,rerank_pool,evidence_per_expert,source_chars,span_width)):
            raise ValueError('Invalid query or budgets')
        if not 1<=k<=candidates<=8 or not candidates<=rerank_pool<=50 or not 1<=evidence_per_expert<=3 or not 1<=source_chars<=6000 or not 1<=span_width<=1200:
            raise ValueError('Invalid query or budgets')
        start=time.perf_counter();retrieved=self.pipeline.search_reranked(query,k=candidates,candidates=rerank_pool,evidence_per_expert=evidence_per_expert)
        rows=retrieved['results'];seen_experts=set();seen_sources=set();proposals={};records=[]
        for i,row in enumerate(rows):
            eid=row['expert']['id'];alias='c'+str(i)
            if eid in seen_experts:raise ValueError('Duplicate expert')
            seen_experts.add(eid)
            for source in row['evidence']:
                if source.get('expert_id')!=eid or source['id'] in seen_sources or not isinstance(source.get('text'),str):raise ValueError('Invalid source ownership')
                seen_sources.add(source['id'])
            pair=propose_pair(query,row['evidence'],width=span_width,source_chars=source_chars)
            if pair is None:continue
            proposals[alias]=pair;records.append({'id':alias,'demand_span':pair['demand_anchor']['quote'],'source_span':pair['source_anchor']['quote']})
        if not records:
            return {'status':'unavailable','reason':'no_lexical_span_pair','results':[],'unresolved':len(rows),
                    'source_snapshot_sha256':retrieved['source_snapshot_sha256'],'seconds':time.perf_counter()-start,
                    'scope':'No model judgment; absent lexical overlap is not expert inability.'}
        questions=[{'type':'choice','name':r['id'],'instructions':FIT_INSTRUCTION+' Evaluate only the selected demand and source spans of '+r['id']+'. The complete need is context, not evidence that the selected source supports other components. Lexical overlap does not prove method fit.',
                    'choices':FIT_CHOICES} for r in records]
        try:
            if self.client is None:
                from laiwenai import LaiwenAI
                self.client=LaiwenAI(model='gpt-6-luna')
            choice=self.decision_fn(self.client,{'complete_need':query,'candidate_span_pairs':records},questions);labels=choice['labels']
            allowed={c['value'] for c in FIT_CHOICES}
            if set(labels)!=set(proposals) or any(v not in allowed for v in labels.values()):raise ValueError('Invalid decision labels')
        except Exception as error:
            return {'status':'failed','error_type':type(error).__name__,'results':[],'unresolved':len(rows),
                    'source_snapshot_sha256':retrieved['source_snapshot_sha256'],'seconds':time.perf_counter()-start,
                    'scope':'No completed model judgment; failure is not expert inability.'}
        rank_order={'direct_application':0,'transferable_method':1,'topic_only':2,'insufficient':3};items=[];judgments=[]
        for i,row in enumerate(rows):
            alias='c'+str(i);eid=row['expert']['id'];pair=proposals.get(alias)
            if pair is None:judgments.append({'expert_id':eid,'status':'unresolved','reason':'no_lexical_span_pair'});continue
            level=labels[alias];judgments.append({'expert_id':eid,'status':'model_choice','model_level':level})
            items.append((i,{'expert':row['expert'],'expert_id':eid,'level':level,'demand_anchor':pair['demand_anchor'],
                             'source_anchor':pair['source_anchor'],'bridge':None,'assumptions':None,
                             'scope':'Lexical anchors and model judgment are unverified; no explanatory bridge, personal role or expert ability is established.'}))
        ranked=[row for _,row in sorted(items,key=lambda item:(rank_order[item[1]['level']],item[0]))][:k]
        return {'status':'complete' if len(ranked)==k else 'partial' if ranked else 'empty','requested':k,'returned':len(ranked),
                'results':ranked,'judgments':judgments,
                'model_positive_returned':sum(r['level'] in ('direct_application','transferable_method') for r in ranked),'unresolved':sum(j['status']=='unresolved' for j in judgments),
                'candidate_ids':[r['expert']['id'] for r in rows],'source_snapshot_sha256':retrieved['source_snapshot_sha256'],
                'retrieval_timing':retrieved['timing'],'decision_seconds':choice.get('seconds'),'seconds':time.perf_counter()-start,
                'scope':'One native choice request on lexically proposed source/demand spans. Neither lexical overlap nor a choice establishes semantic support or a transferable bridge.'}
