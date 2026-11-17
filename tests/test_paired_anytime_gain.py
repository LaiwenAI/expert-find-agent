from itertools import product
import unittest
from expert_search.evaluation_gain import actionable_gain_interval, paired_time_integrated_actionable_gain_interval, time_integrated_gain_interval


class PairedAnytimeTest(unittest.TestCase):
    def test_exhaustive_shared_labels_across_revisions(self):
        first = [(1, ['a', 'b']), (4, ['c', 'a'])]
        second = [(2, ['b', 'c']), (5, ['a'])]
        judgments = {'a': None, 'b': None, 'c': 2}
        result = paired_time_integrated_actionable_gain_interval(first, second, judgments, 6, 2)
        observed = []
        for a, b in product((0, 2), repeat=2):
            labels = dict(judgments, a=a, b=b)
            streams = []
            for events in (first, second):
                streams.append([(t, (x := actionable_gain_interval(r, labels, 2))['lower'], x['upper']) for t, r in events])
            observed.append(time_integrated_gain_interval(streams[0], 6)['lower'] - time_integrated_gain_interval(streams[1], 6)['lower'])
        self.assertAlmostEqual(result['lower'], min(observed))
        self.assertAlmostEqual(result['upper'], max(observed))
        self.assertGreaterEqual(result['lower'], result['independent_bounds'][0])
        self.assertLessEqual(result['upper'], result['independent_bounds'][1])

    def test_identical_trajectories_cancel_unknown(self):
        events = [(1, ['a']), (3, ['b'])]
        result = paired_time_integrated_actionable_gain_interval(events, events, {}, 5)
        self.assertAlmostEqual(result['lower'], 0)
        self.assertAlmostEqual(result['upper'], 0)
        self.assertLess(result['independent_bounds'][0], 0)

    def test_invalid_and_empty_streams(self):
        self.assertEqual(paired_time_integrated_actionable_gain_interval([], [], {}, 1)['upper'], 0)
        with self.assertRaises(ValueError):
            paired_time_integrated_actionable_gain_interval([(2, ['a']), (1, ['b'])], [], {}, 3)
        with self.assertRaises(ValueError):
            paired_time_integrated_actionable_gain_interval([], [], {'x': 5}, 1)


if __name__ == '__main__': unittest.main()
