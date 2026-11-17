import json
import tempfile
import unittest
from pathlib import Path
from expert_search.area_retrieval import AreaSearch
class AreaSearchTest(unittest.TestCase):
    def test_field_scope_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data'
            rows=[{'id':'a:area','expert_id':'a','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u673a\u5668\u4eba\u63a7\u5236\n\u5f3a\u5316\u5b66\u4e60','source':'a:1'},
                  {'id':'b:area','expert_id':'b','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u86cb\u767d\u8d28\n\u7ed3\u6784\u751f\u7269\u5b66','source':'b:1'},
                  {'id':'b:paper','expert_id':'b','kind':'paper','text':'\u673a\u5668\u4eba\u63a7\u5236 \u5f3a\u5316\u5b66\u4e60','source':'b:2'}]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));engine=AreaSearch(p)
            r=engine.search('\u673a\u5668\u4eba\u63a7\u5236 \u5f3a\u5316\u5b66\u4e60')
            self.assertEqual(r['results'][0]['id'],'a');self.assertEqual(r['results'][0]['source'],'a:1')
            self.assertEqual(engine.search('unmatched')['results'],[])
