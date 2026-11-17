import json
import tempfile
import unittest
from pathlib import Path
from expert_search.expert_catalog import ExpertCatalog


class ExpertCatalogTest(unittest.TestCase):
    def test_multibyte_offsets_owner_selection_and_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir()
            rows=[dict(id='s1',expert_id='a',text='\u4e2d\u6587\u8bc1\u636e\uff1a\u56fe\u50cf\u4e0e\u59ff\u6001\u3002'),dict(id='s2',expert_id='b',text='Robot control.'),dict(id='s3',expert_id='a',text='\u7b2c\u4e8c\u6761\uff0c\u8de8\u884c\u5b57\u7b26\n\u4e0d\u6539\u53d8JSON\u884c\u504f\u79fb\u3002')]
            source=corpus/'evidence.jsonl';source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            catalog=ExpertCatalog(corpus,root/'index')
            self.assertEqual(catalog.for_experts(['a','a']),[rows[0],rows[2]])
            self.assertEqual(catalog.for_experts([]),[])
            catalog.db.close()
            rows[0]['text']='changed'
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'snapshot'):ExpertCatalog(corpus,root/'index')
