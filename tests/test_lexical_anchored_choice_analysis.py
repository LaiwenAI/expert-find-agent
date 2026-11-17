import unittest
from experiments.retrieval.analyze_lexical_anchored_choice import summarize


class LexicalChoiceAnalysisTest(unittest.TestCase):
    def test_topic_candidates_do_not_count_as_model_positive(self):
        rows=[{'query_id':'q','status':'complete','ranking':['a','b','c'],
               'levels':['topic_only','transferable_method','insufficient']}]
        labels={'soft':{'q':{'a':1,'b':1,'c':1}}}
        report=summarize(rows,labels)
        self.assertEqual(report['counts']['returned'],3)
        self.assertEqual(report['counts']['model_positive'],1)
        self.assertAlmostEqual(report['model_positive_reviewer_p_at_3']['soft'],1/3)


if __name__=='__main__':unittest.main()
