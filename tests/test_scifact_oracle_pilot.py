import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from experiments.scifact.decisions_oracle_pilot import load


class SciFactArchiveTest(unittest.TestCase):
    def test_load_reads_only_expected_members_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.tar.gz'
            with tarfile.open(path,'w:gz') as archive:
                for name,value in [('data/corpus.jsonl',{'doc_id':1}),('data/claims_dev.jsonl',{'id':2})]:
                    data=(json.dumps(value)+'\n').encode();member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
            corpus,claims=load(path)
            self.assertEqual(corpus,[{'doc_id':1}]);self.assertEqual(claims,[{'id':2}])
            self.assertFalse((Path(directory)/'data').exists())

    def test_duplicate_basename_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.tar.gz'
            with tarfile.open(path,'w:gz') as archive:
                for name in ('data/corpus.jsonl','other/corpus.jsonl','data/claims_dev.jsonl'):
                    member=tarfile.TarInfo(name);member.size=3;archive.addfile(member,io.BytesIO(b'{}\n'))
            with self.assertRaises(ValueError):load(path)


if __name__=='__main__':unittest.main()
