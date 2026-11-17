import itertools
import unittest
from expert_search.attributed_gain import DIMENSIONS, attributed_gain_interval


class AttributedGainTest(unittest.TestCase):
    def test_missing_role_never_becomes_support_or_negative(self):
        result=attributed_gain_interval(['a'],{'a':{'method_support':'supported','temporal_eligibility':'supported'}},k=1)
        self.assertEqual((result['lower'],result['upper']),(0,1))
        self.assertEqual(result['joint_status_counts']['unknown'],1)
        self.assertEqual(result['judged_by_dimension']['personal_contribution'],0)

    def test_bounds_cover_all_unknown_completions_and_are_sharp(self):
        labels={'a':{'method_support':'supported'},'b':{'personal_contribution':'unsupported'},'c':{d:'supported' for d in DIMENSIONS}}
        interval=attributed_gain_interval(['a','b','c'],labels,k=4)
        values=[]
        for first,second in itertools.product(('supported','unsupported'),repeat=2):
            completed={key:dict(value) for key,value in labels.items()}
            completed['a'].update(personal_contribution=first,temporal_eligibility=second)
            result=attributed_gain_interval(['a','b','c'],completed,k=4)
            values.append(result['lower'])
        self.assertAlmostEqual(interval['lower'],min(values))
        self.assertAlmostEqual(interval['upper'],max(values))
        self.assertEqual(interval['returned'],3)

    def test_invalid_inventory_fails_closed(self):
        for ranking,labels in [(['a','a'],{}),(['a'],{'a':{'role':'supported'}}),(['a'],{'a':{'method_support':True}})]:
            with self.assertRaises(ValueError):attributed_gain_interval(ranking,labels)


if __name__=='__main__':unittest.main()
