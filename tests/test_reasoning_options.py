import unittest
from types import SimpleNamespace as NS
from expert_search.agent import ExpertAgent
class ReasoningOptionsTest(unittest.TestCase):
    def test_transport_only_and_usage(self):
        captured={}
        def create(**kwargs):
            captured.update(kwargs)
            return NS(choices=[NS(message=NS(content='{"ok":true}'))],model='fixture',usage=NS(prompt_tokens=2,completion_tokens=1,completion_tokens_details=NS(reasoning_tokens=0)))
        a=ExpertAgent.__new__(ExpertAgent);a.client=NS(chat=NS(completions=NS(create=create)));a.trace=[];a.usage={'prompt_tokens':0,'completion_tokens':0};a.request_options={'reasoning_effort':'none'}
        a.ask('test','Return a judgment',{'evidence':'robot control'})
        self.assertEqual(captured['reasoning_effort'],'none')
        self.assertNotIn('reasoning_effort',captured['messages'][1]['content'])
        self.assertEqual(a.trace[0]['reasoning_tokens'],0)
