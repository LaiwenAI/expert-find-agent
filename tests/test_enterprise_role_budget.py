import importlib.util
from pathlib import Path
import sys
import unittest

experiments=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(experiments))
spec=importlib.util.spec_from_file_location('enterprise_budget',experiments/'enterprise_role_budget.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EnterpriseRoleBudgetTest(unittest.TestCase):
    def test_boundary_crossing_quote_requires_both_fragments(self):
        row=dict(label='supported',sources={'s':dict(text='abcdef')},
                 audits={'reviewer':dict(judgment=dict(citation=dict(evidence_id='s',start=2,end=5,text='cde')))})
        spans=[dict(evidence_id='s',start=0,end=3),dict(evidence_id='s',start=3,end=6),dict(evidence_id='other',start=0,end=6)]
        result=module.citation_retention(row,spans,[0])['reviewer']
        self.assertAlmostEqual(result['fraction'],1/3)
        self.assertFalse(result['complete'])
        self.assertTrue(module.citation_retention(row,spans,[0,1])['reviewer']['complete'])
        self.assertEqual(module.citation_retention(row,spans,[2])['reviewer']['fraction'],0.)
