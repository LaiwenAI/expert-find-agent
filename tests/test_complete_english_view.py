import unittest
from expert_search.bilingual.postvalidation import check_pair
from experiments.bilingual.export_complete_english_view import apply_followup


class CompleteEnglishViewTest(unittest.TestCase):
    def test_followup_remains_unjudged(self):
        row=dict(id='q',english=None,status='unjudged_pair',repair_status='unjudged_repair')
        source=dict(id='q',en='find \u6c34',status='retained_model_screened_pair')
        followup=dict(id='q',prior_status=source['status'],status='unjudged_followup',english='find water',contains_han=False,
                      numeric_check=check_pair(source['en'],'find water'),audit='preserved')
        result=apply_followup(row,source,followup)
        self.assertEqual((result['english'],result['status']),('find water','unjudged_pair'))
        followup['english']='find \u6c34';followup['contains_han']=True
        with self.assertRaises(ValueError):apply_followup(row,source,followup)


if __name__=='__main__':unittest.main()
