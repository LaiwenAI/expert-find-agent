import copy
import unittest
from expert_search.source_counterfactual import build_views,validate_judgments


class SourceCounterfactualTest(unittest.TestCase):
    def fixture(self):
        def source(identifier,title,text,kind='paper'):
            return dict(id=identifier,title=title,text=text,kind=kind,expert_id='e',source=identifier)
        quote='We apply Alpha to Beta in this concrete study.'
        full={'area':source('area','', 'Alpha\nBeta','profile'),
              'p1':source('p1','Alpha for Beta','Alpha for Beta\n'+quote),
              'p2':source('p2','ALPHA for BETA','ALPHA for BETA\n'+quote)}
        reserve=[source('r1','Unrelated topic', 'A different real paper about a different topic. '*3),
                 source('r2','Another topic','Another real record about a separate problem. '*3)]
        demand='Investigate a concrete industrial technical requirement with Alpha and Beta.'
        job=dict(expert_id='e',sources=full,reserve=reserve,demand=demand)
        example=dict(feasible=True,topics=['Alpha','Beta'],relation='Apply Alpha to solve a Beta research problem',
            original_demand_gaps=['No demonstrated industrial delivery'],demand_quote=demand,
            joint_support=dict(evidence_id='p1',quote=quote),independent_support=[
                dict(topic_index=0,evidence_id='area',quote='Alpha'),dict(topic_index=1,evidence_id='area',quote='Beta')])
        return job,example

    def test_alias_closure_and_same_owner_real_replacements(self):
        job,example=self.fixture()
        views=build_views(job,example)
        self.assertEqual(set(views['removed_aliases']),{'p1','p2'})
        self.assertEqual(set(views['control']),{'area','r1','r2'})
        self.assertEqual(len(views['full']),len(views['control']))
        for replacement in views['replacements']:
            self.assertEqual(replacement['original_chars'],replacement['replacement_chars'])
        job['reserve'][0]['expert_id']='other expert'
        with self.assertRaises(ValueError):build_views(job,example)

    def test_no_fabricated_fallback_and_complete_judgments(self):
        job,example=self.fixture()
        bad=copy.deepcopy(job)
        bad['reserve']=[]
        with self.assertRaises(ValueError):build_views(bad,example)
        verdict=dict(judgments=[dict(id=i,label='unknown',reason='Insufficient record') for i in ('a','b','relation')],demand_related=False)
        labels,related=validate_judgments(verdict,job['sources'],job['demand'])
        self.assertEqual(labels['relation'],'unknown')
        self.assertFalse(related)
        verdict['judgments'][2]['id']='a'
        with self.assertRaises(ValueError):validate_judgments(verdict,job['sources'],job['demand'])
