"""One request ranks a supplied pool but explains only the requested top K.

Unreturned candidates remain unassessed, never inferred to lack ability.
"""
import json
import time
from .partial_method_match import PROMPT,build_view,resolve_view
from .partial_match_schema import schema_for_payload


def focused_schema(payload,k):
    n=len(payload['candidates'])
    if type(k) is not int or not 1<=k<=n:raise ValueError('Invalid explanation budget')
    schema=schema_for_payload(payload)
    experts=schema['json_schema']['schema']['properties']['experts']
    experts['minItems']=experts['maxItems']=k
    schema['json_schema']['name']='focused_partial_method_fit'
    return schema


def resolve_focused(value,mapping,k):
    rows=value.get('experts')
    if not isinstance(rows,list) or len(rows)!=k or any(not isinstance(r,dict) for r in rows):raise ValueError('Wrong focused output size')
    ids=[r.get('id') for r in rows]
    if any(not isinstance(i,str) or i not in mapping['experts'] for i in ids) or len(set(ids))!=k:raise ValueError('Duplicate or foreign selected candidate')
    if value.get('read_sources')!=[]:raise ValueError('Raw view already supplied, no extra read allowed')
    subset=dict(mapping,experts={i:mapping['experts'][i] for i in ids})
    result=resolve_view(value,subset)
    result.update(candidate_count=len(mapping['experts']),explained_count=k,
                  unreturned_experts=[eid for name,eid in mapping['experts'].items() if name not in ids],
                  unreturned_status='not_assessed_in_output; no irrelevance or ability judgment')
    return result


class FocusedPartialMethodMatcher:
    def __init__(self,client):self.client=client

    def match(self,demand,candidates,k=3):
        if type(k) is not int or k<1:raise ValueError('Invalid explanation budget')
        start=time.perf_counter();payload,mapping=build_view(demand,candidates,{},'raw');k=min(k,len(candidates))
        prompt=PROMPT.replace('Order experts from most promising to least, include each candidate exactly once.',f'Compare the whole supplied pool. Return ONLY the most promising {k} distinct candidates, in descending partial-method fit order. Generate evidence and explanation only for those selected candidates; do not output tail candidates or infer they lack ability.')
        result=dict(status='unresolved',api_attempts=1)
        try:
            response=self.client.chat.completions.create(messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],reasoning_effort='none',response_format=focused_schema(payload,k),max_completion_tokens=2400)
            result['rounds']=[dict(response=response.model_dump(mode='json'),input_chars=len(json.dumps(payload,ensure_ascii=False)))]
            result.update(result=resolve_focused(json.loads(response.choices[0].message.content),mapping,k),status='structurally_valid')
        except Exception as error:result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        result['wall_seconds']=time.perf_counter()-start
        return result
