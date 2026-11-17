import unittest

from expert_search.owned_role_windows import make_role_record,segment_source,select_owned_windows


class OwnedRoleWindowsTest(unittest.TestCase):
    def test_roles_are_snapshot_bound_and_never_invent_text(self):
        source={'id':'paper','expert_id':'a','text':'abcdefghij'}
        roles={'method':[2],'result':[],'interpretation':[],'background':[0,1],'other':[],'uncertain':[]}
        record=make_role_record(source,roles,4)
        result=select_owned_windows(source,[0,1,2],2,record,4)
        self.assertEqual(result['status'],'role_priority')
        self.assertEqual([w['id'] for w in result['windows']],[2,0])
        for w in result['windows']:
            self.assertEqual(source['text'][w['start']:w['end']],w['text'])
            self.assertEqual(w['expert_id'],'a')
        changed=dict(source,text='abcdefghiX')
        fallback=select_owned_windows(changed,[0,1,2],2,record,4)
        self.assertEqual(fallback['status'],'baseline_invalid_roles')
        self.assertEqual([w['id'] for w in fallback['windows']],[0,1])
        foreign=dict(source,expert_id='b')
        self.assertEqual(select_owned_windows(foreign,[0,1,2],2,record,4)['status'],'baseline_invalid_roles')
        incomplete=dict(roles,method=[2,0],background=[0])
        normalized=make_role_record(source,incomplete,4)
        self.assertEqual(normalized['roles']['uncertain'],[0,1])

    def test_rejects_partial_or_duplicated_baseline_ranking(self):
        source={'id':'p','expert_id':'e','text':'abcde'}
        self.assertEqual(''.join(w['text'] for w in segment_source(source,2)),source['text'])
        with self.assertRaises(ValueError):select_owned_windows(source,[0,0,2],2,width=2)
        with self.assertRaises(ValueError):select_owned_windows(source,[0,1],2,width=2)


if __name__=='__main__':unittest.main()
