import json
import tempfile
import unittest
from pathlib import Path
from expert_search.area_retrieval import AreaSearch
class AreaSearchTest(unittest.TestCase):
    def test_field_scope_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data'
            rows=[{'id':'a:area','expert_id':'a','field':'研究领域_CN','text':'机器人控制\n强化学习','source':'a:1'},
                  {'id':'b:area','expert_id':'b','field':'研究领域_CN','text':'蛋白质\n结构生物学','source':'b:1'},
                  {'id':'b:paper','expert_id':'b','kind':'paper','text':'机器人控制 强化学习','source':'b:2'}]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));engine=AreaSearch(p)
            r=engine.search('机器人控制 强化学习')
            self.assertEqual(r['results'][0]['id'],'a');self.assertEqual(r['results'][0]['source'],'a:1')
            self.assertEqual(engine.search('unmatched')['results'],[])
