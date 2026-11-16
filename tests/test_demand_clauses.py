import unittest
from expert_search.demand_evidence_data import validate_clauses


class ClausesTest(unittest.TestCase):
    def test_source_offsets_and_background(self):
        background='目前产业技术系统仍存在多种问题，需要进一步解决并开展深入研究。'
        requirement='开发机器人运动控制算法，支持多传感器输入并实现实时轨迹规划和碰撞检测。'
        text='机器人技术系统研发需求标题足够长以避免被长度校验自动拒绝\n（一）命题背景 '+background+' （二）命题内容 '+requirement
        demand={'query':text}
        start,end,quote=validate_clauses(demand,[requirement])[0]
        self.assertEqual(text[start:end],quote)
        for invalid in (background,text.split('\n')[0],requirement.replace('实时','快速')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_clauses(demand,[invalid])
        with self.assertRaises(ValueError):validate_clauses(demand,[requirement,requirement])
