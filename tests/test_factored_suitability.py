import unittest
from expert_search.factored_suitability import combine, questions


class FactoredSuitabilityTest(unittest.TestCase):
    def test_three_valued_conjunction(self):
        self.assertEqual(combine(dict(core='demonstrated',breadth='demonstrated')),1)
        for a,b in [('not_demonstrated','demonstrated'),('uncertain','not_demonstrated'),('not_demonstrated','uncertain')]:
            self.assertEqual(combine(dict(core=a,breadth=b)),0)
        self.assertIsNone(combine(dict(core='demonstrated',breadth='uncertain')))
        self.assertIsNone(combine(dict(core='uncertain',breadth='uncertain')))

    def test_missing_or_foreign_answers_rejected(self):
        for value in [{},{'core':'demonstrated'},{'core':'yes','breadth':'demonstrated'}]:
            with self.assertRaises(ValueError):combine(value)

    def test_predicates_require_core_and_distinct_skills(self):
        value=questions('RULE')
        self.assertEqual([q['name'] for q in value],['core','breadth'])
        self.assertTrue(all(q['instructions'].startswith('RULE') for q in value))
        self.assertIn('TWO DISTINCT',value[1]['instructions'])
