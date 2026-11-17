import copy
import unittest
from expert_search.operation_plan import validate_plan


class OperationPlanTest(unittest.TestCase):
    def setUp(self):
        self.text = '\u8bc6\u522b\u59ff\u6001\u5e76\u90e8\u7f72\u5230\u7ec8\u7aef\u3002'
        self.refs = {'s': dict(start=0, end=len(self.text), text=self.text)}
        self.plan = dict(complete=True, operations=[dict(id='a', object='\u7528\u6237\u59ff\u6001', operation='\u8bc6\u522b', qualifiers=[], span_ids=['s'])],
                         obligations=[dict(id='b', kind='deployment', text='\u90e8\u7f72\u5230\u7ec8\u7aef', operation_ids=['a'], span_ids=['s'])])

    def test_system_obligation_is_separate_from_component(self):
        result = validate_plan(self.text, self.refs, self.plan)
        self.assertEqual(len(result['operations']), 1)
        self.assertEqual(result['obligations'][0]['operation_ids'], ['a'])

    def test_duplicate_operations_and_unsupported_dependencies_rejected(self):
        duplicate = copy.deepcopy(self.plan)
        duplicate['operations'].append(dict(duplicate['operations'][0], id='c'))
        with self.assertRaisesRegex(ValueError, 'duplicate operation'):
            validate_plan(self.text, self.refs, duplicate)
        changed = copy.deepcopy(self.plan)
        changed['obligations'][0]['operation_ids'] = ['not-present']
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            validate_plan(self.text, self.refs, changed)

    def test_partial_or_changed_source_cannot_pass_as_complete(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_plan(self.text, self.refs, dict(self.plan, complete=False))
        with self.assertRaisesRegex(ValueError, 'Changed'):
            validate_plan('\u4e0d\u540c\u539f\u6587', self.refs, self.plan)
