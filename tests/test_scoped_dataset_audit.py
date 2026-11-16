import copy
import unittest
from expert_search.scoped_dataset_audit import validate_verdict


class ScopedAuditTest(unittest.TestCase):
    def fixture(self):
        demand='Concrete industrial requirement '+('x'*30)
        sources={'s':dict(text='Unique alpha and beta topic record.',source='file:1')}
        verdict=dict(requirements=[dict(id=cid,label='supported',evidence_id='s',quote=quote,reason='Direct topic')
                                   for cid,quote in [('c0','alpha'),('c1','beta')]],
            replacement=dict(label='unknown',reason='No supporting record'),demand_related=True,
            shared_technical_topic='alpha',demand_quote=demand)
        return verdict,['alpha','beta'],sources,demand

    def test_every_condition_and_requirement_span_are_bound(self):
        verdict,conditions,sources,demand=self.fixture()
        result=validate_verdict(verdict,conditions,sources,demand)
        self.assertTrue(result['passes'])
        self.assertEqual(set(result['requirements']),{'c0','c1'})
        for mutate in ('omit','duplicate','quote','demand'):
            bad=copy.deepcopy(verdict)
            if mutate=='omit':bad['requirements'].pop()
            if mutate=='duplicate':bad['requirements'][1]['id']='c0'
            if mutate=='quote':bad['requirements'][0]['quote']='not present'
            if mutate=='demand':bad['demand_quote']='not present'*3
            with self.assertRaises(ValueError):validate_verdict(bad,conditions,sources,demand)
        with self.assertRaises(ValueError):validate_verdict(verdict,[],sources,demand)

    def test_unknown_and_unrelated_are_not_positive_or_inability(self):
        verdict,conditions,sources,demand=self.fixture()
        verdict['requirements'][0]['label']='unknown'
        self.assertFalse(validate_verdict(verdict,conditions,sources,demand)['passes'])
        verdict['demand_related']=False
        verdict['demand_quote']=''
        self.assertFalse(validate_verdict(verdict,conditions,sources,demand)['passes'])
