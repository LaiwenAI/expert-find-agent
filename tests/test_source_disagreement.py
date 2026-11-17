import unittest
from experiments.bilingual.analyze_source_disagreement import decompose


class SourceDisagreementTest(unittest.TestCase):
    def test_shared_and_unique_decomposition(self):
        z=dict(ranking=[str(i) for i in range(10)],source_kinds=['profile']*10)
        e=dict(ranking=[str(i) for i in range(5)]+[str(i) for i in range(10,15)],source_kinds=['paper']*10)
        result=decompose(z,e)
        self.assertEqual(result['common_experts'],5)
        self.assertEqual(result['transitions']['profile->paper'],5)
        self.assertEqual(result['common_contribution'],.5)
        self.assertEqual(result['unique_contribution'],.5)
        with self.assertRaises(ValueError):decompose(z,dict(ranking=['x']*10,source_kinds=['paper']*10))


if __name__=='__main__':unittest.main()
