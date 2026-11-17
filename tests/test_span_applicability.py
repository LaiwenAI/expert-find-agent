import unittest

from expert_search.span_applicability import source_spans, resolve_span_assessments


class SpanApplicabilityTest(unittest.TestCase):
    def setUp(self):
        self.sources = {'s': dict(expert_id='e', text='\u76f8\u540c\u539f\u6587\u76f8\u540c\u539f\u6587')}
        self.spans = source_spans('e', self.sources, width=4)
        self.row = dict(requirement_id='r', level='topic_only', span_ids=['p1'], assumptions=[])

    def test_repeated_text_has_unambiguous_unicode_offsets(self):
        result = resolve_span_assessments('e', {'r': {}}, self.sources, self.spans, dict(assessments=[self.row]))
        citation = result['assessments']['r']['citations'][0]
        self.assertEqual(citation, dict(evidence_id='s', start=4, end=8, quote='\u76f8\u540c\u539f\u6587'))

    def test_unknown_duplicate_tampered_and_foreign_spans_are_rejected(self):
        for ids in (['missing'], ['p0', 'p0']):
            with self.assertRaises(ValueError):
                resolve_span_assessments('e', {'r': {}}, self.sources, self.spans, dict(assessments=[dict(self.row, span_ids=ids)]))
        for changed in (dict(self.spans['p1'], quote='\u6539\u5199\u5f15\u6587'), dict(self.spans['p1'], evidence_id='foreign')):
            with self.assertRaises(ValueError):
                resolve_span_assessments('e', {'r': {}}, self.sources, dict(p1=changed), dict(assessments=[self.row]))
        with self.assertRaises(ValueError):
            source_spans('other', self.sources)
