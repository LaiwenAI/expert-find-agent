import unittest
from collections import Counter
from experiments.bilingual.analyze_unflagged_identity_review import validate_rows,weighted_intervals


class IdentityReviewTest(unittest.TestCase):
    def test_bounds_preserve_uncertain_and_unsampled(self):
        counts={'a':Counter(yes=1,no=1,uncertain=1,missing=1),'b':Counter(no=2)}
        result=weighted_intervals(counts,{'a':100,'b':50})
        self.assertAlmostEqual(result['weighted_sample_bounds'][0],1/6)
        self.assertAlmostEqual(result['weighted_sample_bounds'][1],.5)
        self.assertEqual(result['finite_population_worst_case'],[1/150,147/150])

    def test_input_immutability_and_reviewer(self):
        base=[dict(case_id='c',english_request='A request.',specific_identity=None,indirect_identifiers=None,notes=None,reviewer_id=None)]
        valid=[dict(base[0],specific_identity='no',reviewer_id='reviewer')]
        self.assertEqual(validate_rows(base,valid)['c']['specific_identity'],'no')
        with self.assertRaises(ValueError):validate_rows(base,[dict(valid[0],english_request='Changed')])
        with self.assertRaises(ValueError):validate_rows(base,[dict(valid[0],reviewer_id=None)])
        with self.assertRaises(ValueError):validate_rows(base,[dict(valid[0],specific_identity='unknown')])


if __name__=='__main__':unittest.main()
