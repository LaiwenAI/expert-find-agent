import tempfile
import unittest
from expert_search.decision_cache import DecisionCache
class CacheTest(unittest.TestCase):
    def test_content_and_model_version(self):
        calls=[]
        def call(prompt,rows):
            calls.append(rows)
            return {'judgments':[{'id':r['id'],'label':'supported','quote':r['text']} for r in rows]}
        with tempfile.TemporaryDirectory() as d:
            pairs=[{'constraint':'robot','text':'robot control'}]*3
            c=DecisionCache(d+'/cache','model-a',call)
            self.assertEqual(c.judge(pairs)['unique_misses'],1)
            self.assertEqual(c.judge(pairs)['network_calls'],0)
            self.assertEqual(DecisionCache(d+'/cache','model-b',call).judge(pairs)['network_calls'],1)
            pairs[0]={'constraint':'robot','text':'new robot evidence'}
            self.assertEqual(c.judge(pairs)['unique_misses'],1)
