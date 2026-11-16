import unittest

from expert_search.source_applicability import resolve_assessments


class SourceApplicabilityTest(unittest.TestCase):
    def setUp(self):
        self.sources = {'s': dict(expert_id='e', text='Documented pose estimation method.')}
        self.row = dict(requirement_id='r', level='transferable_method',
                        citations=[dict(evidence_id='s', quote='pose estimation')],
                        transfer_reason='Pose estimates could inform rehabilitation movement analysis.',
                        assumptions=['Validate clinical movement angles on target patients.'])

    def test_transfer_preserves_validation_gap_and_exact_span(self):
        result = resolve_assessments('e', {'r': {}}, self.sources, dict(assessments=[self.row]))
        self.assertEqual(result['direct_coverage'], 0)
        self.assertEqual(result['transfer_coverage'], 1)
        self.assertEqual(result['unresolved_requirements'], ['r'])
        self.assertEqual(result['assessments']['r']['citations'][0]['start'], 11)

    def test_missing_requirement_is_not_silently_scored_as_negative(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            resolve_assessments('e', {'r': {}, 'r2': {}}, self.sources, dict(assessments=[self.row]))

    def test_paraphrase_and_foreign_owner_are_rejected(self):
        for sources, quote in ((self.sources, 'human pose prediction'),
                               ({'s': dict(expert_id='other', text=self.sources['s']['text'])}, 'pose estimation')):
            row = dict(self.row, citations=[dict(evidence_id='s', quote=quote)])
            with self.assertRaises(ValueError):
                resolve_assessments('e', {'r': {}}, sources, dict(assessments=[row]))
