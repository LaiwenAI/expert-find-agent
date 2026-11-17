"""Basic invariants of the lexical overlap screen."""
import unittest
from experiments.retrieval.audit_cof_near_overlap import shingles


class ShingleTests(unittest.TestCase):
    def test_case_and_punctuation_are_normalized(self):
        self.assertEqual(shingles('One, TWO three four five six.'),shingles('one two three four five six'))
        self.assertEqual(len(shingles('one two three four five six')),2)

    def test_short_text_has_no_five_word_shingles(self):
        self.assertEqual(shingles('one two three four'),set())


if __name__=='__main__':unittest.main()
