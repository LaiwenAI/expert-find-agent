import unittest
from expert_search.witness_planner import independent_inventory,assemble


class WitnessPlannerTest(unittest.TestCase):
    def fixture(self):
        def paper(eid,title,text):return dict(id=eid,title=title,text=text,kind='paper',expert_id='e')
        papers=[paper('j','Alpha methods for beta tasks','Alpha methods for beta tasks with an explicit documented relationship.'),
            paper('copy','A different record','  ALPHA methods for beta TASKS with other formatting'),
            paper('a','Alpha foundations','A separate alpha foundation study with sufficient source length.'),
            paper('b','Beta foundations','A separate beta foundation study with sufficient source length.'),
            paper('f','Other work','Another paper with enough characters for real replacement controls.'),
            paper('r','Reserve work','A real reserve paper with enough characters for matched replacements, plus details.')]
        job=dict(id='job',expert_id='e',demand='A real technical requirement involving alpha methods and beta tasks.',
            sources={s['id']:s for s in papers[:4]},reserve=papers[4:],selection=dict(paper_cosines={s['id']:1. for s in papers}))
        plan=dict(feasible=True,topics=['alpha methods','beta tasks'],relation='Alpha methods applied to beta tasks.',
            joint_support=dict(evidence_id='j',quote='Alpha methods for beta tasks'),
            demand_quote='technical requirement involving alpha methods and beta tasks',original_demand_gaps=['deployment unknown'])
        return job,plan

    def test_shadow_copies_removed_before_retrieval_and_control_assembly(self):
        job,plan=self.fixture()
        joint,remaining=independent_inventory(job,plan)
        self.assertEqual(joint['id'],'j');self.assertNotIn('copy',remaining)
        witnesses=[dict(topic_index=0,evidence_id='a',quote='A separate alpha foundation study'),
                   dict(topic_index=1,evidence_id='b',quote='A separate beta foundation study')]
        prepared,example,views=assemble(job,plan,witnesses,set(remaining))
        self.assertNotIn('copy',prepared['sources'])
        self.assertNotIn('j',views['control'])
        self.assertTrue({'a','b'}<=set(views['control']))
        self.assertEqual(len(views['full']),len(views['control']))
        with self.assertRaises(ValueError):assemble(job,plan,witnesses,{'a'})

    def test_owner_context_alias_and_explicit_length_gate(self):
        job,plan=self.fixture()
        with self.assertRaises(ValueError):independent_inventory(job,dict(plan,relation='x'*241))
        job['reserve'][0]['expert_id']='other'
        with self.assertRaises(ValueError):independent_inventory(job,plan)
        job,plan=self.fixture()
        job['sources']['field']=dict(id='field',kind='profile',expert_id='e',text='Alpha methods for beta tasks')
        with self.assertRaises(ValueError):independent_inventory(job,plan)
