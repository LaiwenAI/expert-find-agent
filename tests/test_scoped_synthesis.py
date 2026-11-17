import unittest
from expert_search.synthesis_v2 import render_scoped_queries


class RenderingTest(unittest.TestCase):
    def test_one_capability_changed_and_no_extra_application(self):
        base,changed=render_scoped_queries([{'text':'\u673a\u5668\u4eba\u63a7\u5236'},{'text':'\u8def\u5f84\u89c4\u5212'}],
                                         {'changed_index':1,'replacement_text':'\u56fa\u4f53\u5e9f\u5f03\u7269\u5904\u7406'})
        self.assertEqual(base.replace('\u8def\u5f84\u89c4\u5212','\u56fa\u4f53\u5e9f\u5f03\u7269\u5904\u7406'),changed)
        self.assertIn('\u4e0d\u8981\u6c42\u8054\u5408\u5e94\u7528',base)
        for index in (True,-1,2):
            with self.assertRaises(ValueError):
                render_scoped_queries([{'text':'A'},{'text':'B'}],{'changed_index':index,'replacement_text':'C'})
