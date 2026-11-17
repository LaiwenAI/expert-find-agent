"""Deadline replay must use the latest result, including a worse revision."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.retrieval.cof_progressive_deadlines import analyze


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class ProgressiveDeadlineTest(unittest.TestCase):
    def test_later_harmful_revision_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';root.mkdir();pilot=Path(tmp)/'pilot';pilot.mkdir()
            data=root/'labels.json';data.write_text('data')
            (pilot/'protocol.json').write_text(json.dumps({'query_ids':['q'],'input_sha256':{'labels.json':digest(data)}}))
            (pilot/'candidate-snapshots.json').write_text('{}')
            rows=[{'query_id':'q','stage':'screen','retrieval_seconds':0.1,'screen_seconds':0.9,'baseline':['good'],'screened':['bad']}]
            (pilot/'results.json').write_text(json.dumps(rows))
            summary={key:digest(pilot/name) for name,key in [('protocol.json','protocol_sha256'),('results.json','results_sha256'),('candidate-snapshots.json','candidate_snapshots_sha256')]}
            (pilot/'summary.json').write_text(json.dumps(summary))
            labels={'soft':{'q':{'good':1,'bad':0}}}
            with patch('experiments.retrieval.cof_progressive_deadlines.load',return_value=(root,{}, {'good':{},'bad':{}},labels)):
                result=analyze(pilot,deadlines=(0.05,0.5,1.,2.))
            self.assertEqual([r['p_at_5']['soft']['progressive'] for r in result['deadlines']],[0.,0.2,0.,0.])
            self.assertEqual(result['deadlines'][2]['screen_available'],1)
            self.assertAlmostEqual(result['time_average_p_at_5']['soft']['progressive'],0.09)


if __name__=='__main__':unittest.main()
