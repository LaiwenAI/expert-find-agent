import unittest
from expert_search.demand_evidence_data import validate_clauses


class ClausesTest(unittest.TestCase):
    def test_source_offsets_and_background(self):
        background='\u76ee\u524d\u4ea7\u4e1a\u6280\u672f\u7cfb\u7edf\u4ecd\u5b58\u5728\u591a\u79cd\u95ee\u9898\uff0c\u9700\u8981\u8fdb\u4e00\u6b65\u89e3\u51b3\u5e76\u5f00\u5c55\u6df1\u5165\u7814\u7a76\u3002'
        requirement='\u5f00\u53d1\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236\u7b97\u6cd5\uff0c\u652f\u6301\u591a\u4f20\u611f\u5668\u8f93\u5165\u5e76\u5b9e\u73b0\u5b9e\u65f6\u8f68\u8ff9\u89c4\u5212\u548c\u78b0\u649e\u68c0\u6d4b\u3002'
        text='\u673a\u5668\u4eba\u6280\u672f\u7cfb\u7edf\u7814\u53d1\u9700\u6c42\u6807\u9898\u8db3\u591f\u957f\u4ee5\u907f\u514d\u88ab\u957f\u5ea6\u6821\u9a8c\u81ea\u52a8\u62d2\u7edd\n\uff08\u4e00\uff09\u547d\u9898\u80cc\u666f '+background+' \uff08\u4e8c\uff09\u547d\u9898\u5185\u5bb9 '+requirement
        demand={'query':text}
        start,end,quote=validate_clauses(demand,[requirement])[0]
        self.assertEqual(text[start:end],quote)
        for invalid in (background,text.split('\n')[0],requirement.replace('\u5b9e\u65f6','\u5feb\u901f')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_clauses(demand,[invalid])
        with self.assertRaises(ValueError):validate_clauses(demand,[requirement,requirement])
