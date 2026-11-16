import unittest
from expert_search.batch_applicability import batch_inputs,resolve_compact_batch


class BatchApplicabilityTest(unittest.TestCase):
    def setUp(self):
        self.mapping,_=batch_inputs({'a':{'s1':dict(expert_id='a',text='Pose method.')},'b':{'s2':dict(expert_id='b',text='Other method.')}})

    def test_foreign_candidate_span_is_rejected_without_losing_other_candidate(self):
        response=dict(experts=[dict(id='c0',assessments=[dict(r='r',l='o',s=['c1:p0'])]),dict(id='c1',assessments=[dict(r='r',l='u')])])
        result=resolve_compact_batch({'r':{}},self.mapping,response)
        self.assertEqual(result['a']['status'],'unresolved')
        self.assertEqual(result['b']['status'],'structurally_valid')

    def test_missing_requirement_or_missing_candidate_is_not_silently_unknown(self):
        response=dict(experts=[dict(id='c0',assessments=[]),dict(id='c1',assessments=[dict(r='r',l='u')])])
        result=resolve_compact_batch({'r':{}},self.mapping,response)
        self.assertEqual(result['a']['status'],'unresolved')
        with self.assertRaises(ValueError):resolve_compact_batch({'r':{}},self.mapping,dict(experts=response['experts'][:1]))

    def test_transfer_requires_mechanism_assumptions_and_owned_source(self):
        response=dict(experts=[dict(id='c0',assessments=[dict(r='r',l='t',s=['c0:p0'])]),dict(id='c1',assessments=[dict(r='r',l='u')])])
        self.assertEqual(resolve_compact_batch({'r':{}},self.mapping,response)['a']['status'],'unresolved')
