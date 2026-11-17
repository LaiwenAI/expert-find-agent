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
            self.assertEqual(catalog.for_evidence(['s3','s1','s3','missing']),[rows[0],rows[2]])
            self.assertEqual(catalog.for_evidence([]),[])
            with self.assertRaises(ValueError):catalog.for_evidence([''])
            catalog.db.close()
            rows[0]['text']='changed'
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'snapshot'):ExpertCatalog(corpus,root/'index')

    def test_open_catalog_rejects_source_and_database_mutations(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir();source=corpus/'evidence.jsonl'
            source.write_text(json.dumps(dict(id='s',expert_id='a',text='old'))+'\n')
            catalog=ExpertCatalog(corpus,root/'index')
            source.write_text(json.dumps(dict(id='s',expert_id='a',text='new'))+'\n')
            with self.assertRaisesRegex(ValueError,'snapshot'):catalog.for_evidence(['s'])
            catalog.db.close()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir();source=corpus/'evidence.jsonl'
            source.write_text(json.dumps(dict(id='s',expert_id='a',text='old'))+'\n')
            catalog=ExpertCatalog(corpus,root/'index')
            with sqlite3.connect(catalog.database) as db:db.execute("UPDATE evidence SET expert_id='b' WHERE id='s'")
            with self.assertRaisesRegex(ValueError,'snapshot'):catalog.for_evidence(['s'])
            catalog.db.close()

    def test_reopen_rejects_catalog_preexisting_mutation(self):
        import sqlite3
        for change in ("DELETE FROM evidence WHERE id='s'", "UPDATE evidence SET expert_id='b' WHERE id='s'", "UPDATE evidence SET offset=offset+1 WHERE id='s'"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);corpus=root/'corpus';corpus.mkdir()
                (corpus/'evidence.jsonl').write_text(json.dumps(dict(id='s',expert_id='a',text='old'))+'\n')
                catalog=ExpertCatalog(corpus,root/'index');database=catalog.database;catalog.db.close()
                with sqlite3.connect(database) as db:db.execute(change)
                with self.assertRaisesRegex(ValueError,'inventory'):ExpertCatalog(corpus,root/'index')

    def test_open_catalog_rejects_database_file_replacement(self):
        import os
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir()
            (corpus/'evidence.jsonl').write_text(json.dumps(dict(id='s',expert_id='a',text='text'))+'\n')
            catalog=ExpertCatalog(corpus,root/'index')
            replacement=root/'replacement.sqlite';shutil.copyfile(catalog.database,replacement);os.replace(replacement,catalog.database)
            with self.assertRaisesRegex(ValueError,'snapshot'):catalog.for_evidence(['s'])
            catalog.db.close()
