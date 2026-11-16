import itertools
import unittest
from expert_search.query_lift import query_lift,compare_lift,comparison_coefficients,plan_comparison_judgments


class LiftTest(unittest.TestCase):
    def test_static_prior_zero_despite_high_precision_and_missing_labels(self):
        run={'q1':['popular'],'q2':['popular']};labels={'q1':{'popular':1},'q2':{'popular':1}}
        result=query_lift(run,labels,k=1)
        self.assertEqual(result['observed_gain']['lower'],1)
        self.assertEqual(result['lower'],0)
        self.assertEqual(query_lift(run,{},k=1)['upper'],0)

    def test_exact_permutation_expectation_and_replication(self):
        run={'q1':['a'],'q2':['b'],'q3':['c']}
        labels={q:{e:int(q=='q'+str(i+1)) for i,e in enumerate('abc')} for q in run}
        result=query_lift(run,labels,k=1)
        expectations=[]
        for perm in itertools.permutations(run):
            expectations.append(sum(labels[source][run[target][0]] for target,source in zip(run,perm))/3)
        self.assertAlmostEqual(result['query_shuffle_gain']['lower'],sum(expectations)/len(expectations))
        self.assertAlmostEqual(result['lower'],2/3)
        doubled={q+suffix:list(r) for suffix in ('x','y') for q,r in run.items()}
        doubled_labels={q+suffix:labels[q] for suffix in ('x','y') for q in run}
        self.assertAlmostEqual(query_lift(doubled,doubled_labels,k=1)['lower'],result['lower'])

    def test_missing_label_bounds_are_sharp_and_shared(self):
        run={'q1':['a'],'q2':['b']};known={'q1':{'a':1}}
        result=query_lift(run,known,k=1);completions=[]
        for values in itertools.product((0,1),repeat=3):
            labels={'q1':{'a':1,'b':values[0]},'q2':{'a':values[1],'b':values[2]}}
            completions.append(query_lift(run,labels,k=1)['lower'])
        self.assertAlmostEqual(result['lower'],min(completions))
        self.assertAlmostEqual(result['upper'],max(completions))
        self.assertEqual(compare_lift(run,run,{},k=1)['lower'],0)
        self.assertEqual(compare_lift(run,run,{},k=1)['upper'],0)

    def test_unknown_states_not_silently_numeric(self):
        with self.assertRaises(ValueError):query_lift({'q1':['a'],'q2':['b']},{'q1':{'a':'unknown'}},k=1)

    def test_annotation_plan_is_optimal_for_linear_box_width(self):
        first={'q1':['a'],'q2':['b'],'q3':['a']}
        second={'q1':['b'],'q2':['b'],'q3':['b']}
        for metric in ('lift','gain'):
            coefficients=comparison_coefficients(first,second,k=1,metric=metric)
            plan=plan_comparison_judgments(first,second,{},2,k=1,metric=metric)
            widths=[sum(abs(value) for cell,value in coefficients.items() if cell not in chosen)
                    for chosen in itertools.combinations(coefficients,2)]
            self.assertAlmostEqual(plan['minimum_remaining_width_if_all_resolved'],min(widths))
            revealed={}
            for request in plan['requests']:
                revealed.setdefault(request['query_id'],{})[request['expert_id']]=1
            next_plan=plan_comparison_judgments(first,second,revealed,0,k=1,metric=metric)
            self.assertAlmostEqual(next_plan['bounds']['upper']-next_plan['bounds']['lower'],min(widths))
        self.assertEqual(plan_comparison_judgments(first,first,{},100,k=1)['requests'],[])
