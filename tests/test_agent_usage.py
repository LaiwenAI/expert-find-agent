import types
import unittest
from expert_search.agent import ExpertAgent


class AgentUsageTests(unittest.TestCase):
    def test_missing_usage_is_recorded_as_missing(self):
        agent=ExpertAgent.__new__(ExpertAgent)
        message=types.SimpleNamespace(content='{}')
        response=types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],usage=None,model='test')
        agent.client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs: response)))
        agent.trace=[];agent.usage={'prompt_tokens':0,'completion_tokens':0}
        agent.ask('test','instruction',{})
        self.assertFalse(agent.trace[0]['usage_present'])
        self.assertEqual(agent.usage['prompt_tokens'],0)
