import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RetrievedCoverageReviewTest(unittest.TestCase):
    def test_export_uses_first_valid_model_citation_and_visible_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();data=root/'artifacts/dataset-release/fresh-enterprise-eval-v1';data.mkdir(parents=True)
            query=(json.dumps({'id':'q','split':'development','text':'need'})+'\n').encode();(data/'queries.jsonl').write_bytes(query)
            (run/'protocol.json').write_text(json.dumps({'query_ids':['q'],'arms':['direct','coverage_direct'],'query_file_sha256':hashlib.sha256(query).hexdigest()}))
            hits=[{'id':'expert','evidence':[{'id':'source','expert_id':'expert','text':'x'*7000}]}]
            raw=json.dumps(hits,sort_keys=True).encode();(run/'q-pool.json').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            row={'query_id':'q','arm':'direct','input_sha256':digest,'status':'ok','ranking':['expert'],
                 'trace':[{'output':{'results':[{'expert_id':'expert','citations':['bad']},{'expert_id':'expert','citations':['source']}]}}]}
            (run/'q-direct.json').write_text(json.dumps(row));row.update(arm='coverage_direct',status='error')
            (run/'q-coverage_direct.json').write_text(json.dumps(row))
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+str(Path(__file__).resolve().parents[1]/'src'))
            subprocess.run([sys.executable,'-m','experiments.retrieval.export_retrieved_coverage_review','run','out'],cwd=root,env=env,check=True,capture_output=True)
            form=json.loads((root/'out/blind-review.jsonl').read_text())
            self.assertEqual(form['evidence'][0]['text'],'x'*6000)
            self.assertNotIn('arm',form);self.assertNotIn('expert',form)


if __name__=='__main__':unittest.main()
