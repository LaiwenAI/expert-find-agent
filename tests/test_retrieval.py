import json
import tempfile
import unittest
from pathlib import Path
from expert_search.retrieval import Search,index,tokens

class RetrievalTest(unittest.TestCase):
    def test_ranking_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,rows in {'experts':[{'id':'a'},{'id':'b'}], 'evidence':[
                {'id':'e1','expert_id':'a','text':'机器人 robot control','source':'source:1'},
                {'id':'e2','expert_id':'b','text':'protein chemistry','source':'source:2'}]}.items():
                (p/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            index(p,p/'db'); engine=Search(p,p/'db')
            result=engine.search('机器人 control')
            self.assertEqual(result[0]['id'],'a')
            self.assertEqual(result[0]['evidence'][0]['source'],'source:1')
            self.assertEqual(engine.search('robot',exclude=['e1']),[])
            self.assertEqual(engine.search('unmatched'),[])
    def test_chinese(self):
        self.assertEqual(tokens('机器人 Robot'),['机器','器人','robot'])

if __name__=='__main__': unittest.main()
