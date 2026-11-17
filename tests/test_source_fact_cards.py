from types import SimpleNamespace
import unittest
from expert_search.source_fact_cards import compile_source_fact_cards


class Response:
    def __init__(self,content):self.choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    def model_dump(self,**kwargs):return {'choices':[{'message':{'content':self.choices[0].message.content}}]}


class SourceFactCardTest(unittest.TestCase):
    def client(self,content):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:Response(content))))

    def test_empty_extraction_and_profile_method_claims_keep_raw_fallback(self):
        source=dict(id='s',expert_id='e',kind='profile',text='Research interests include machine learning.')
        empty=compile_source_fact_cards(source,self.client('{"facts":[]}'))
        self.assertEqual(empty['status'],'raw_fallback')
        invalid=compile_source_fact_cards(source,self.client('{"facts":[{"id":"f0","kind":"method","claim":"A documented research method.","span_ids":["p0"]}]}'))
        self.assertEqual(invalid['status'],'raw_fallback')
        self.assertEqual(invalid['cards'],[])
