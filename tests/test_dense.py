import tempfile
import unittest
from types import SimpleNamespace as NS
from expert_search.dense_retrieval import encode,DenseSearch
class DenseTest(unittest.TestCase):
    def test_resume_and_ranking(self):
        calls=[]
        def create(input):
            calls.append(input)
            texts=[input] if isinstance(input,str) else input
            return NS(model='frozen',data=[NS(index=i,embedding=[1.,0.] if t=='robot' else [0.,1.]) for i,t in enumerate(texts)])
        client=NS(embeddings=NS(create=create));texts=[{'id':'a','text':'robot'},{'id':'b','text':'biology'}]
        with tempfile.TemporaryDirectory() as d:
            encode(texts,d,client,batch_size=2);n=len(calls)
            encode(texts,d,client,batch_size=2);self.assertEqual(len(calls),n)
            self.assertEqual(DenseSearch(d,client).search('robot',1)[0]['id'],'a')
            with self.assertRaises(ValueError):encode(texts,d,client,batch_size=1)
            with self.assertRaises(ValueError):encode([{'id':'a','text':'changed'}],d,client)
