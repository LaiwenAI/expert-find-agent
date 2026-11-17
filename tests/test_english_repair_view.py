import unittest
from expert_search.bilingual.postvalidation import check_pair
from experiments.bilingual.export_english_repair_view import build_row


class EnglishRepairViewTest(unittest.TestCase):
    def test_retained_requires_both_checks_and_previous_status(self):
        source=dict(id='q',parent_id='p',intent_type='overview',en='find \u6c34 flow',status='retained_model_screened_pair')
        repair=dict(id='q',prior_status=source['status'],english='find water flow',contains_han=False,
                    numeric_check=check_pair(source['en'],'find water flow'),audit='preserved',status='model_screened_repair')
        self.assertEqual(build_row(source,repair)['status'],'retained_model_screened_pair')
        source['status']='unjudged_pair';repair['prior_status']='unjudged_pair';repair['status']='unjudged_repair'
        self.assertEqual(build_row(source,repair)['status'],'unjudged_pair')
        repair['status']='model_screened_repair'
        with self.assertRaises(ValueError):build_row(source,repair)

    def test_remaining_han_is_not_exported(self):
        source=dict(id='q',parent_id='p',intent_type='overview',en='find \u6c34',status='unjudged_pair')
        repair=dict(id='q',prior_status='unjudged_pair',english='seek \u6c34',contains_han=True,
                    numeric_check=check_pair(source['en'],'seek \u6c34'),status='unjudged_repair')
        self.assertIsNone(build_row(source,repair)['english'])
        with self.assertRaises(ValueError):build_row(source)


if __name__=='__main__':unittest.main()
