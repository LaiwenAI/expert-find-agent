import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from expert_search.agent import ExpertAgent
from expert_search.retrieval import index

class AgentTest(unittest.TestCase):
    def test_exclusion_and_citation_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'experts.jsonl').write_text(json.dumps({'id':'a','name':'A'}))
            (p/'evidence.jsonl').write_text('\n'.join(json.dumps(x) for x in [
                {'id':'keep','expert_id':'a','kind':'paper','text':'robot control'},
                {'id':'heldout','expert_id':'a','kind':'paper','text':'robot secret'}]))
            index(p,p/'db')
            outputs=iter([{'requirements':['robot'],'queries':[]},{'leaf_ids':['heldout','keep','fake']},
                {'results':[{'expert_id':'a','citations':['heldout','keep','fake'],'reason':'supported'}]}])
            def create(**kw):
                payload=json.loads(kw['messages'][1]['content'])
                if 'trees' in payload: self.assertNotIn('heldout',json.dumps(payload['trees']))
                return NS(choices=[NS(message=NS(content=json.dumps(next(outputs))))],usage=None)
            client=NS(chat=NS(completions=NS(create=create)))
            result=ExpertAgent(p,p/'db',client).run('robot',1,1,exclude={'heldout'})
            self.assertEqual([x['id'] for x in result['results'][0]['evidence']],['keep'])
