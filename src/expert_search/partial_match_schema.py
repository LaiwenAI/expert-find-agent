"""Native output schemas bound references to each candidate's supplied view."""
import json
from types import SimpleNamespace
from .partial_method_match import PartialMethodMatcher


def text(low,high):
    return dict(type='string',pattern=r'^[\s\S]{'+str(low)+','+str(high)+'}$')


def array(items,low,high):return dict(type='array',items=items,minItems=low,maxItems=high)


def ids(values,low,high):
    return array(dict(type='string',enum=list(values)) if values else dict(type='string'),low,high)


def object_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def schema_for_payload(payload):
    variants=[];unread=[];demand=list(payload['demand_spans']);definitions={}
    def shared_ids(values,name,low,high):
        if not values:return ids(values,low,high)
        definitions[name]=dict(type='string',enum=list(values))
        return array({'$ref':'#/$defs/'+name},low,high)
    for candidate in payload['candidates']:
        refs=[];methods=[]
        for source in candidate['sources']:
            if not source['raw_supplied']:unread.append(source['id'])
            for fact in source['facts']:
                refs.append(fact['id'])
                if fact['kind']=='method':methods.append(fact['id'])
            for span in source.get('raw_spans',[]):refs.append(span['id']);methods.append(span['id'])
        for level in ['d','t','o','u']:
            if level in ('d','t') and not methods:continue
            if level=='o' and not refs:continue
            unknown=level=='u';actionable=level in ('d','t')
            variants.append(object_schema(dict(id=dict(type='string',enum=[candidate['id']]),l=dict(type='string',enum=[level]),
                d=shared_ids(demand,'demand_id',0 if unknown else 1,0 if unknown else 2),
                e=shared_ids(methods if actionable else refs,candidate['id']+('_method_ref' if actionable else '_ref'),0 if unknown else 1,0 if unknown else 2),
                m=text(1 if level=='t' else 0,0 if unknown else 120),
                a=array(text(1,80),1 if actionable else 0,1 if actionable else 0),g=array(text(1,100),1,1))))
    schema=object_schema(dict(experts=array(dict(anyOf=variants),len(payload['candidates']),len(payload['candidates'])),read_sources=ids(unread,0,len(unread))))
    schema['$defs']=definitions
    return dict(type='json_schema',json_schema=dict(name='partial_method_fit',strict=True,schema=schema))


class StructuredPartialMethodMatcher(PartialMethodMatcher):
    def __init__(self,client):
        def create(**kwargs):
            payload=json.loads(kwargs['messages'][-1]['content'])
            kwargs['response_format']=schema_for_payload(payload)
            return client.chat.completions.create(**kwargs)
        super().__init__(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
