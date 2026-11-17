import unittest
from expert_search.partial_method_match import build_view
from expert_search.focused_partial_match import focused_schema,resolve_focused


class FocusedPartialTest(unittest.TestCase):
    def setUp(self):
        candidates={e:{e+'s':dict(id=e+'s',expert_id=e,text='Own method evidence.',kind='paper')} for e in ['a','b','c']}
        self.payload,self.mapping=build_view('Demand operation.',candidates,{},'raw')
        self.row=dict(id='c1',l='t',d=['d0'],e=['c1:s0:p0'],m='Method can support target operation.',a=['Target data requires validation.'],g=['Integration remains unshown.'])

    def test_budget_selects_subset_and_preserves_unknown_tail(self):
        s=focused_schema(self.payload,1)['json_schema']['schema']['properties']['experts']
        self.assertEqual((s['minItems'],s['maxItems']),(1,1))
        r=resolve_focused(dict(experts=[self.row],read_sources=[]),self.mapping,1)
        self.assertEqual(r['ranking'][0]['expert_id'],'b')
        self.assertEqual(r['unreturned_experts'],['a','c'])
        self.assertIn('not_assessed',r['unreturned_status'])

    def test_subset_still_rejects_borrowed_evidence_and_duplicate_candidates(self):
        row=dict(self.row,e=['c0:s0:p0'])
        with self.assertRaises(ValueError):resolve_focused(dict(experts=[row],read_sources=[]),self.mapping,1)
        with self.assertRaises(ValueError):resolve_focused(dict(experts=[self.row,self.row],read_sources=[]),self.mapping,2)
        with self.assertRaises(ValueError):focused_schema(self.payload,0)
