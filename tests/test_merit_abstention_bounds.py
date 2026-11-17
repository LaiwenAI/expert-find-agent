from itertools import product
import unittest
from experiments.analyze_merit_suitability import metrics


class AbstentionBoundsTest(unittest.TestCase):
    def test_bounds_cover_exact_extrema_of_all_missing_binary_completions(self):
        labels=[1,0,1,0,1,0];partial=[1,1,0,0,None,None]
        bounds=metrics(labels,partial)['full_cohort_bounds'];scores={k:[] for k in bounds}
        for a,b in product([0,1],repeat=2):
            predicted=[1,1,0,0,a,b]
            tp=sum(y==p==1 for y,p in zip(labels,predicted));fp=sum(y==0 and p==1 for y,p in zip(labels,predicted));fn=sum(y==1 and p==0 for y,p in zip(labels,predicted));tn=sum(y==p==0 for y,p in zip(labels,predicted))
            values=dict(accuracy=(tp+tn)/6,precision=tp/(tp+fp),recall=tp/3,f1=2*tp/(2*tp+fp+fn),balanced_accuracy=(tp/3+tn/3)/2)
            for k,v in values.items():scores[k].append(v)
        for key,values in scores.items():
            self.assertAlmostEqual(bounds[key][0],min(values),msg=key)
            self.assertAlmostEqual(bounds[key][1],max(values),msg=key)
        self.assertEqual(partial,[1,1,0,0,None,None])
