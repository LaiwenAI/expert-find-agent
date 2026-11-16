import json
import tempfile
import unittest
from pathlib import Path
from expert_search.cmu_baseline import run
class GoldBaselineTest(unittest.TestCase):
    def test_self_paper_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ds=root/'evaluation_datasets/d_20_1';(ds/'archives').mkdir(parents=True);(root/'data').mkdir()
            (root/'data/evaluations.csv').write_text('ParticipantID\tPaper1\tPaper2\tExpertise1\tExpertise2\nr\ta\tb\t4.5\t1\n')
            a={'id':'a','content':{'title':'robot control','abstract':''}}
            b={'id':'b','content':{'title':'protein biology','abstract':''}}
            c={'id':'c','content':{'title':'robot manipulation','abstract':''}}
            (ds/'submissions.json').write_text(json.dumps({'a':a,'b':b}))
            (ds/'archives/~r.jsonl').write_text('\n'.join(json.dumps(x) for x in [a,c]))
            r=run(root,root/'results')
            self.assertEqual(r['self_paper_matches_excluded'],1)
            self.assertEqual(r['weighted_pairwise_loss'],0)
