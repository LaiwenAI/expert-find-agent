import itertools
import unittest
from expert_search.source_binding import binding_witness, record_key, phrase_index


class SourceBindingTest(unittest.TestCase):
    def test_joint_and_split_are_not_interchangeable(self):
        self.assertEqual(binding_witness([{'x'}, {'x'}], 'same_record'), ['x', 'x'])
        self.assertIsNone(binding_witness([{'x'}, {'x'}], 'distinct_records'))
        self.assertIsNone(binding_witness([{'x'}, {'y'}], 'same_record'))
        self.assertEqual(binding_witness([{'x'}, {'y'}], 'distinct_records'), ['x', 'y'])

    def test_matching_against_exhaustive_enumeration(self):
        universe = {'x', 'y', 'z'}
        sets = [set(x for i,x in enumerate(sorted(universe)) if bits & (1<<i)) for bits in range(8)]
        for facets in itertools.product(sets, repeat=3):
            expected = any(all(p[i] in facets[i] for i in range(3)) for p in itertools.permutations(universe))
            witness = binding_witness(facets, 'distinct_records')
            self.assertEqual(witness is not None, expected)
            if witness:
                self.assertEqual(len(set(witness)), 3)
                self.assertTrue(all(w in f for w, f in zip(witness, facets)))

    def test_duplicate_titles_do_not_create_independence(self):
        self.assertEqual(record_key('Alpha  BETA'), record_key('alpha beta'))
        postings, _, edges = phrase_index([
            dict(id='p1', expert_id='e', title='机器学习图像识别'),
            dict(id='p2', expert_id='e', title='机器学习图像识别')])
        first, second = postings['机器学习']['e'], postings['图像识别']['e']
        self.assertIsNone(binding_witness([first, second], 'distinct_records'))
        self.assertEqual(len(edges['e', next(iter(first))]), 2)
