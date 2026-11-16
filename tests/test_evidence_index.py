import json
import tempfile
import unittest
from pathlib import Path
from expert_search.evidence_index import build,EvidenceIndex
class EvidenceIndexTest(unittest.TestCase):
    def test_spans_and_expert_filter(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r={'id':'e','expert_id':'a','text':'Research in robot control systems.\nIndependent biomedical project data.','source':'fixture:1'}
            (p/'data').write_text(json.dumps(r))
            build(p/'data',p/'db');i=EvidenceIndex(p/'db')
            hits=i.lookup('robot control',['a'],1)['a'];self.assertEqual(len(hits),1)
            h=hits[0];self.assertEqual(h['text'],r['text'][h['start']:h['end']])
            self.assertEqual(i.lookup('robot',['b'],1),{'b':[]})
