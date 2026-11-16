import unittest
from expert_search.synthesis_v2 import render_scoped_queries


class RenderingTest(unittest.TestCase):
    def test_one_capability_changed_and_no_extra_application(self):
        base,changed=render_scoped_queries([{'text':'机器人控制'},{'text':'路径规划'}],
                                         {'changed_index':1,'replacement_text':'固体废弃物处理'})
        self.assertEqual(base.replace('路径规划','固体废弃物处理'),changed)
        self.assertIn('不要求联合应用',base)
        for index in (True,-1,2):
            with self.assertRaises(ValueError):
                render_scoped_queries([{'text':'A'},{'text':'B'}],{'changed_index':index,'replacement_text':'C'})
