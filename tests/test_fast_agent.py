import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from expert_search.fast_agent import FastExpertAgent
class FastAgentTest(unittest.TestCase):
    def test_explicit_conjunction_and_explanation_citation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'experts.jsonl').write_text('\n'.join(json.dumps({'id':x,'name':x}) for x in ['a','b']))
            rows=[{'id':'e-a','expert_id':'a','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u673a\u5668\u4eba\n\u4eba\u5de5\u667a\u80fd','source':'a:1'},
                {'id':'e-b','expert_id':'b','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u673a\u5668\u4eba','source':'b:1'}]
            (p/'evidence.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            answer={'experts':[{'id':'a','citation_id':'e-a','reason':'supported'},{'id':'b','citation_id':'fake'}]}
            client=NS(chat=NS(completions=NS(create=lambda **kw:NS(choices=[NS(message=NS(content=json.dumps(answer)))],usage=None))))
            events=list(FastExpertAgent(p,client).events('\u5bfb\u627e\u300c\u673a\u5668\u4eba\u300d\u548c\u300c\u4eba\u5de5\u667a\u80fd\u300d\u4e13\u5bb6',2,True))
            self.assertEqual([r['expert']['id'] for r in events[0]['results']],['a'])
            self.assertEqual([r['id'] for r in events[1]['experts']],['a'])
            self.assertEqual(events[0]['stage'],'retrieval');self.assertEqual(events[1]['stage'],'explanation')

    def test_full_stage_replaces_initial_context(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'experts.jsonl').write_text('\n'.join(json.dumps({'id':x,'name':x}) for x in ['a','b']))
            rows=[{'id':'e-a','expert_id':'a','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u673a\u5668\u4eba','source':'a:1'}, {'id':'e-b','expert_id':'b','field':'\u7814\u7a76\u9886\u57df_CN','text':'\u751f\u7269\u5b66','source':'b:1'}]
            (p/'evidence.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            def create(**kwargs):
                payload=json.loads(kwargs['messages'][1]['content'])
                self.assertEqual([r['expert']['id'] for r in payload['results']],['b'])
                return NS(choices=[NS(message=NS(content=json.dumps({'experts':[{'id':'b','citation_id':'paper-b','reason':'paper evidence'}]})))],usage=None)
            client=NS(chat=NS(completions=NS(create=create)))
            agent=FastExpertAgent(p,client)
            agent.full=NS(search=lambda *args:[{'id':'b','score':2,'evidence':[{'id':'paper-b','text':'robot paper','source':'b:2'}]}])
            events=list(agent.events('\u673a\u5668\u4eba',1,True))
            self.assertEqual([r['stage'] for r in events],['retrieval','full_evidence','explanation'])
            self.assertEqual(events[-1]['experts'][0]['id'],'b')
