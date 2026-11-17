import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.openalex.build_author_recovery import record
from experiments.openalex.verify_author_recovery import verify


class OpenAlexAuthorRecoveryTest(unittest.TestCase):
    def test_frozen_data_and_tampering(self):
        root=Path('benchmarks/openalex-author-recovery-v1')
        result=verify(root);self.assertGreater(result['queries'],0)
        with tempfile.TemporaryDirectory() as temporary:
            copied=Path(temporary)/'data';shutil.copytree(root,copied)
            with (copied/'qrels.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaises(ValueError):verify(copied)

    def test_rejects_wrong_language_or_future_history(self):
        row={'id':'https://openalex.org/W1','title':'Robot control','publication_year':2019,'language':'en',
             'authorships':[{'author':{'id':'https://openalex.org/A1'}}]}
        self.assertEqual(record(row,'history')['authors'],['A1'])
        self.assertIsNone(record(dict(row,language='zh'),'history'))
        self.assertIsNone(record(dict(row,publication_year=2021),'history'))
        self.assertIsNone(record(dict(row,title='Robot \u673a\u5668\u4eba'),'history'))


if __name__=='__main__':unittest.main()
