import unittest
from expert_search.span_view_applicability import SpanViewApplicability, validate_span_view


class SpanViewTest(unittest.TestCase):
    def test_foreign_owner_and_rewritten_excerpt_rejected_before_request(self):
        sources={'s':dict(expert_id='a',text='abcdef')}
        correct=dict(evidence_id='s',start=1,end=3,quote='bc')
        validate_span_view('a',sources,{'p':correct})
        for expert,span in [('b',correct),('a',dict(correct,quote='ab')),('a',dict(correct,end=99))]:
            with self.assertRaises(ValueError):
                SpanViewApplicability(client=object()).assess_view('d',expert,{'r':dict(text='r')},sources,{'p':span})
