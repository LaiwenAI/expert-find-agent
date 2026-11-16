import unittest
from expert_search.synthesis import validate
class SynthesisTest(unittest.TestCase):
    def test_quotes_and_unknown(self):
        source={'a':{'text':'research on robot control'}}
        e={'query':'robotics','constraints':[{'text':'robot control','label':'supported','evidence_id':'a','quote':'robot control'}]}
        self.assertEqual(validate(e,source),[])
        e['constraints'][0]['quote']='quantum robot'
        self.assertTrue(validate(e,source))
        e['constraints'][0]['label']='unknown'
        self.assertEqual(validate(e,source),[])
