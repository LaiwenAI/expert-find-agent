"""Training-free fast retrieval and reusable batched constraint verification."""
import time
from .agent import ExpertAgent
from .fast_retrieval import FastSearch
from .decision_cache import DecisionCache
from .constraint_metrics import evidence_gain


class ConstraintAgent:
    def __init__(self, model='gpt-6-luna', cache='artifacts/constraint-decisions.sqlite'):
        from laiwenai import LaiwenAI
        self.model=ExpertAgent(client=LaiwenAI(model=model))
        self.search=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
        self.cache=DecisionCache(cache,model,lambda instruction,payload:self.model.ask('verify_batch',instruction,payload),batch_size=32)

    def run(self,query,k=5,candidates=10,evidence_mode="document",fixed_constraints=None):
        if not query.strip() or not 1<=k<=candidates<=50:raise ValueError('Invalid query or candidate budget')
        start=time.perf_counter();self.model.trace=[]
        plan={'constraints':fixed_constraints} if fixed_constraints is not None else self.model.ask('constraints','Extract at most 4 distinct atomic technical constraints from the demand, preserving its conditions. Return {"constraints":[{"text":str,"mandatory":bool}]}. Do not invent requirements.',{'demand':query})
        constraints=plan.get('constraints',[])
        if not isinstance(constraints,list) or not 1<=len(constraints)<=4 or any(not isinstance(c,dict) or not isinstance(c.get('text'),str) or not c['text'].strip() or not isinstance(c.get('mandatory'),bool) for c in constraints):
            raise ValueError('Malformed constraints')
        recalled=self.search.search(query,candidates,'bm25-vote')
        pairs=[];owners=[]
        if evidence_mode not in ['document','segment']:raise ValueError('Unknown evidence mode')
        segment_hits={}
        if evidence_mode=='segment':
            from .evidence_index import EvidenceIndex
            idx=EvidenceIndex()
            for j,c in enumerate(constraints):
                segment_hits[j]=idx.lookup(c['text'],[e['id'] for e in recalled],2)
        for expert in recalled:
            for j,c in enumerate(constraints):
                selected=expert['evidence'][:2] if evidence_mode=='document' else segment_hits[j][expert['id']]
                for evidence in selected:
                    evidence=dict(evidence)
                    if 'id' not in evidence:evidence['id']=evidence['evidence_id']
                    pairs.append({'constraint':c['text'],'text':evidence['text'][:4000]})
                    owners.append((expert['id'],str(j),evidence))
        judged=self.cache.judge(pairs);by_expert={e['id']:{} for e in recalled};citations={e['id']:[] for e in recalled}
        # Contradiction is retained even if a different source supports the constraint.
        conflicts={e['id']:[] for e in recalled}
        for (eid,j,evidence),answer in zip(owners,judged['judgments']):
            old=by_expert[eid].get(j,'unknown');label=answer['label']
            if {old,label}=={'supported','contradicted'}:conflicts[eid].append(j)
            if label=='contradicted' or (label=='supported' and old!='contradicted'):by_expert[eid][j]=label
            else:by_expert[eid].setdefault(j,old)
            if label!='unknown':citations[eid].append(dict(constraint_id=j,label=label,quote=answer['quote'],evidence_id=evidence['id'],source=evidence['source']))
        schema={str(j):dict(weight=1,mandatory=c['mandatory']) for j,c in enumerate(constraints)}
        ranked=[]
        for expert in recalled:
            gain=evidence_gain(schema,by_expert[expert['id']])
            ranked.append(dict(expert=expert,judgments=by_expert[expert['id']],conflicts=sorted(set(conflicts[expert['id']])),citations=citations[expert['id']],**gain))
        ranked.sort(key=lambda r:(-r['gated_gain'],-r['supported_coverage'],-r['expert']['score'],r['expert']['id']))
        return dict(query=query,evidence_mode=evidence_mode,input_characters=sum(len(p['text']) for p in pairs),constraints=constraints,results=ranked[:k],seconds=time.perf_counter()-start,
            cache_stats={key:value for key,value in judged.items() if key!='judgments'},trace=self.model.trace,
            limitations='Model judgments are not gold; only two recalled evidence items per candidate are checked; no global completeness guarantee.')
