import json
import tempfile
import unittest
from pathlib import Path
from expert_search.expert_catalog import ExpertCatalog


class ExpertCatalogTest(unittest.TestCase):
    def test_multibyte_offsets_owner_selection_and_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir()
            rows=[dict(id='s1',expert_id='a',text='中文证据：图像与姿态。'),dict(id='s2',expert_id='b',text='Robot control.'),dict(id='s3',expert_id='a',text='第二条，跨行字符\n不改变JSON行偏移。')]
            source=corpus/'evidence.jsonl';source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            catalog=ExpertCatalog(corpus,root/'index')
            self.assertEqual(catalog.for_experts(['a','a']),[rows[0],rows[2]])
            self.assertEqual(catalog.for_experts([]),[])
            catalog.db.close()
            rows[0]['text']='changed'
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'snapshot'):ExpertCatalog(corpus,root/'index')
