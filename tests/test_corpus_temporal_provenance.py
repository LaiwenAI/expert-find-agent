import json
import tempfile
import unittest
from pathlib import Path
from expert_search.corpus import build


class CorpusTemporalProvenanceTest(unittest.TestCase):
    def test_paper_metadata_retains_partial_date_and_unverified_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'input'/'teachers'/'shenzhen';root.mkdir(parents=True)
            (root/'teachers.jsonl').write_text(json.dumps({'teacher_id':'t1'})+'\n')
            (root/'papers.jsonl').write_text(json.dumps({'teacher_id':'t1','title':'Paper','gs_publication_date':'2020'})+'\n')
            output=Path(tmp)/'output';build(Path(tmp)/'input',output)
            paper=json.loads((output/'evidence.jsonl').read_text())
            self.assertEqual(paper['publication']['precision'],'year')
            self.assertEqual(paper['publication']['start'],'2020-01-01')
            self.assertEqual(paper['contribution_status'],'unverified')
            self.assertEqual(paper['authorship_association'],'provided_teacher_id_not_independently_verified')


if __name__=='__main__':unittest.main()
