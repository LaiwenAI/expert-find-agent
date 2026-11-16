import unittest
from expert_search.evidence_packing import pack,idf_weights


class PackingTest(unittest.TestCase):
    def test_budget_sources_and_determinism(self):
        docs=[dict(id='a',text='neural retrieval '+('background '*100),similarity=.9),
              dict(id='b',text='bayesian personalization '+('context '*100),similarity=.8)]
        weights=idf_weights(d['text'] for d in docs)
        result=pack(docs,'neural retrieval bayesian personalization',weights,budget=120,window_size=60)
        self.assertLessEqual(sum(len(w['text']) for w in result),120)
        self.assertEqual(result,pack(list(reversed(docs)),'neural retrieval bayesian personalization',weights,budget=120,window_size=60))
        for window in result:
            original=next(d['text'] for d in docs if d['id']==window['paper_id'])
            self.assertEqual(original[window['start']:window['end']],window['text'])

    def test_coverage_avoids_redundant_documents(self):
        docs=[dict(id='a',text='neural retrieval',similarity=.9),dict(id='b',text='neural retrieval',similarity=.89),
              dict(id='c',text='bayesian personalization',similarity=.8)]
        weights=idf_weights(d['text'] for d in docs)
        result=pack(docs,'neural retrieval bayesian personalization',weights,budget=41,window_size=24)
        self.assertEqual({r['paper_id'] for r in result},{'a','c'})
        self.assertEqual({r['paper_id'] for r in pack(docs,'neural retrieval bayesian personalization',weights,budget=32,window_size=24,mode='similarity')},{'a','b'})
