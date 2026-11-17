"""Provenance guards must reject stale benchmark and candidate corpora."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.retrieval.cof_progressive_screen_pilot import verify_prior


class CoFScreenProvenanceTest(unittest.TestCase):
    def test_rejects_changed_sources_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'benchmark'; root.mkdir()
            prior = Path(tmp)/'prior'; (prior/'corpus').mkdir(parents=True)
            names = ('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')
            for name in names: (root/name).write_text(name)
            hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
            (prior/'summary.json').write_text(json.dumps({'input_sha256':hashes}))
            (prior/'corpus/experts.jsonl').write_text(json.dumps({'id':'reviewer'})+'\n')
            source = {'id':json.dumps(['reviewer','paper'],separators=(',',':')),'expert_id':'reviewer','paper_id':'paper','text':'Title\nAbstract'}
            path = prior/'corpus/evidence.jsonl'; path.write_text(json.dumps(source)+'\n')
            papers = {'paper':{'title':'Title','abstract':'Abstract'}}
            reviewers = {'reviewer':{'papers':['paper']}}
            self.assertEqual(verify_prior(root,papers,reviewers,prior),hashes)
            path.write_text(json.dumps(dict(source,text='Changed'))+'\n')
            with self.assertRaisesRegex(ValueError,'evidence corpus'): verify_prior(root,papers,reviewers,prior)
            path.write_text(json.dumps(source)+'\n')
            (root/names[0]).write_text('Changed')
            with self.assertRaisesRegex(ValueError,'different benchmark inputs'): verify_prior(root,papers,reviewers,prior)


if __name__ == '__main__': unittest.main()
