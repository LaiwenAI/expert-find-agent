import unittest
from expert_search.grouped_dataset import split_components


def row(i,demand,expert,source):
    return dict(id=i,parent_demand_id=demand,constructing_expert_id=expert,query=i+' base',
                counterfactual_query=i+' changed',references=[dict(source_snapshot_sha256=source)])


class GroupedDatasetTest(unittest.TestCase):
    def test_transitive_links_cannot_cross_splits(self):
        rows=[row('1','d1','e1','s1'),row('2','d2','e1','s2'),row('3','d2','e3','s3'),
              row('4','d4','e4','s4'),row('5','d5','e5','s5'),row('6','d6','e6','s6')]
        result=split_components(rows)
        self.assertTrue(result['feasible'])
        owner={i:name for name,ids in result['assignments'].items() for i in ids}
        self.assertEqual(len({owner[i] for i in ('1','2','3')}),1)
        self.assertEqual(result,split_components(rows))
        self.assertTrue(all(result['counts'].values()))

    def test_identical_query_or_source_prevents_fake_independence(self):
        first=row('1','d1','e1','s1')
        second=row('2','d2','e2','s2')
        second['counterfactual_query']=first['query'].upper()
        third=row('3','d3','e3','s2')
        result=split_components([first,second,third])
        self.assertFalse(result['feasible'])
        self.assertEqual(sorted(result['components'][0]),['1','2','3'])
