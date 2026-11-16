import math
import unittest
from expert_search.expert_language_models import ExpertLanguageModels


class LanguageModelsTest(unittest.TestCase):
    def test_probabilities_against_manual_marginalization(self):
        model = ExpertLanguageModels({'d1':'a a b','d2':'b c c'}, {'e1':['d1'],'e2':['d2'],'both':['d1','d2']})
        baseline = (1/6)**2
        first = model.score('a c','model1-document')
        second = model.score('a c','model2-document')
        self.assertAlmostEqual(first['both'], math.log((1/9)/baseline))
        self.assertAlmostEqual(first['e1'], math.log((1/12)/baseline))
        self.assertAlmostEqual(second['both'], math.log((1/12)/baseline))
        self.assertAlmostEqual(second['e1'], second['e2'])

    def test_single_token_linearity_and_long_query_stability(self):
        model = ExpertLanguageModels({'x':'alpha beta','y':'beta gamma'}, {'e':['x','y'],'f':['y']})
        a,b = model.score('alpha'), model.score('alpha','model2-document')
        for eid in a:
            self.assertAlmostEqual(a[eid],b[eid])
        self.assertTrue(all(math.isfinite(v) for v in model.score('alpha '*10000,'model2-candidate').values()))

    def test_empty_records_and_unknown_source_auditable(self):
        model = ExpertLanguageModels({'x':'alpha','empty':''}, {'e':['x','empty']})
        self.assertEqual(model.empty_sources,['empty'])
        self.assertEqual(model.members['e'],[0])
        with self.assertRaises(ValueError):
            ExpertLanguageModels({'x':'alpha'}, {'e':['absent']})
