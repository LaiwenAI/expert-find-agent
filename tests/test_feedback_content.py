import unittest
from expert_search.feedback_content import Correction, apply_corrections


class FeedbackContentTests(unittest.TestCase):
    def test_text_changes_ranking_only_when_scope_and_source_match(self):
        ranking=['a','b','c']; sources={'b':{'p':'A verified quote in a paper.'}}
        valid=Correction('b','battery','p','verified quote','promote')
        self.assertEqual(apply_corrections('battery recycling',ranking,[valid],sources)['ranking'],['b','a','c'])
        changed=Correction('b','solar','p','verified quote','promote')
        self.assertEqual(apply_corrections('battery recycling',ranking,[changed],sources)['ranking'],ranking)
        swapped=Correction('b','battery','p','unrelated quote','promote')
        self.assertEqual(apply_corrections('battery recycling',ranking,[swapped],sources)['audit'],['unanchored'])

    def test_conflicting_corrections_abstain(self):
        sources={'b':{'p':'quoted'}}
        corrections=[Correction('b','query','p','quoted',a) for a in ('promote','demote')]
        result=apply_corrections('query',['a','b'],corrections,sources)
        self.assertEqual(result['ranking'],['a','b'])
        self.assertEqual(result['conflicts'],['b'])

    def test_missing_owner_or_non_candidate_abstains(self):
        correction=Correction('b','query','p','quoted','promote')
        self.assertEqual(apply_corrections('query',['a'],[correction],{'b':{'p':'quoted'}})['audit'],['unanchored'])
        self.assertEqual(apply_corrections('query',['a','b'],[correction],{'a':{'p':'quoted'}})['audit'],['unanchored'])

if __name__=='__main__': unittest.main()
