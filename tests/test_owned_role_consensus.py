import unittest

from expert_search.owned_role_consensus import agree_role_records
from expert_search.owned_role_windows import make_role_record


class OwnedRoleConsensusTest(unittest.TestCase):
    def test_disagreement_is_unknown_and_stale_source_is_rejected(self):
        source={'id':'p','expert_id':'e','text':'abcdefgh'}
        first={'method':[0],'result':[1],'interpretation':[],'background':[],'other':[],'uncertain':[]}
        second={'method':[0],'result':[],'interpretation':[],'background':[1],'other':[],'uncertain':[]}
        a=make_role_record(source,first,4);b=make_role_record(source,second,4)
        result=agree_role_records(source,a,b,4)
        self.assertEqual(result['roles']['method'],[0]);self.assertEqual(result['roles']['uncertain'],[1]);self.assertEqual(result['disagreements'],1)
        with self.assertRaises(ValueError):agree_role_records(dict(source,text='abcdXfgh'),a,b,4)


if __name__=='__main__':unittest.main()
