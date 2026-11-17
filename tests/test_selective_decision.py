import unittest
from unittest.mock import patch
from expert_search.selective_decision import selective_decision


QUESTION = {'name': 'fit', 'choices': [{'value': 'yes'}, {'value': 'no'}]}


def native(score, label='yes'):
    return {'labels': {'fit': label}, 'response': {'answers': [{'probabilities': [
        {'value': 'yes', 'probability': score}, {'value': 'no', 'probability': 1-score}]}]}}


class SelectiveDecisionTests(unittest.TestCase):
    @patch('expert_search.selective_decision.decide')
    def test_threshold_boundary_and_escalation(self, decide):
        decide.return_value = native(.75)
        calls = []
        def slow(records, question):
            calls.append((records, question))
            return 'no'
        records = {'evidence': 'example'}
        accepted = selective_decision(None, records, QUESTION, slow, .75)
        self.assertEqual((accepted['label'], accepted['escalated']), ('yes', False))
        self.assertEqual(calls, [])
        escalated = selective_decision(None, records, QUESTION, slow, .76)
        self.assertEqual((escalated['label'], escalated['escalated']), ('no', True))
        self.assertEqual(calls, [(records, QUESTION)])

    @patch('expert_search.selective_decision.decide')
    def test_failures_do_not_invent_or_reuse_labels(self, decide):
        decide.side_effect = RuntimeError('native')
        result = selective_decision(None, {}, QUESTION, lambda *_: 'yes', .75)
        self.assertEqual((result['status'], result['label'], result['failure_stage']), ('unresolved', None, 'native'))
        decide.side_effect = None; decide.return_value = native(.6)
        result = selective_decision(None, {}, QUESTION, lambda *_: 'foreign', .75)
        self.assertEqual((result['status'], result['label'], result['failure_stage']), ('unresolved', None, 'slow'))
        self.assertTrue(result['escalated'])

    def test_invalid_configuration(self):
        for value in (float('nan'), -1, 2, True):
            with self.assertRaises(ValueError):
                selective_decision(None, {}, QUESTION, lambda *_: 'yes', value)


if __name__ == '__main__': unittest.main()
