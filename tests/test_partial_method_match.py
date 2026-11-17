import hashlib
import unittest
from expert_search.partial_method_match import build_view,resolve_view


class PartialMethodTest(unittest.TestCase):
    def setUp(self):
        self.source=dict(id='s',expert_id='a',text='A documented research method.',kind='paper')
        self.candidates={'a':{'s':self.source},'b':{'z':dict(id='z',expert_id='b',text='Another source.',kind='paper')}}

    def test_foreign_card_and_cross_candidate_reference_rejected(self):
        card=dict(source_id='s',expert_id='wrong',source_sha256=hashlib.sha256(self.source['text'].encode()).hexdigest(),cards=[])
        with self.assertRaises(ValueError):build_view('Demand operation',self.candidates,{'s':card},'cards')
        _,mapping=build_view('Demand operation',self.candidates,{},'raw')
        rows=[dict(id='c0',l='t',d=['d0'],e=['c1:s0:p0'],m='A functional connection.',a=['Requires validation.'],g=['Other project requirements unknown.']),dict(id='c1',l='u',d=[],e=[],m='',a=[],g=['Unknown.'])]
        with self.assertRaises(ValueError):resolve_view(dict(experts=rows,read_sources=[]),mapping)

    def test_missing_cards_preserve_raw_context_and_unknown_is_not_a_negative(self):
        payload,mapping=build_view('Demand operation',self.candidates,{},'cards')
        self.assertTrue(all(s['raw_supplied'] for c in payload['candidates'] for s in c['sources']))
        rows=[dict(id=c,l='u',d=[],e=[],m='',a=[],g=['Missing support in supplied snapshot.']) for c in mapping['experts']]
        r=resolve_view(dict(experts=rows,read_sources=[]),mapping)
        self.assertEqual([a['level'] for a in r['ranking']],['insufficient','insufficient'])
        self.assertTrue(all(a['scope']=='partial_method_fit_only' for a in r['ranking']))
