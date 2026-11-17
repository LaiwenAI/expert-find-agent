import unittest

from experiments.bilingual.analyze_alignment_dependence import analyze


class AlignmentDependenceTest(unittest.TestCase):
    def test_quartiles_and_cluster_bootstrap(self):
        rows = [dict(id=str(i), group_id=str(i // 2), query_cosine=float(i),
                     language_overlap={'exact': i / 10}) for i in range(8)]
        report = analyze(rows)
        self.assertEqual(report['groups'], 4)
        self.assertEqual([b['count'] for b in report['quartiles']], [2, 2, 2, 2])
        self.assertAlmostEqual(report['pearson_r'], 1)
        self.assertAlmostEqual(report['quartiles'][0]['mean_top10_overlap'], .05)
        self.assertAlmostEqual(report['quartiles'][-1]['mean_top10_overlap'], .65)

    def test_rejects_duplicate_ids(self):
        row = dict(id='same', group_id='a', query_cosine=.8, language_overlap={'exact': .5})
        with self.assertRaises(ValueError):
            analyze([row] * 4)


if __name__ == '__main__':
    unittest.main()
