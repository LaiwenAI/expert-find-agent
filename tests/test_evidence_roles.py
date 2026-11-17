import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('evidence_roles', Path(__file__).resolve().parents[1] / 'experiments/evidencebench_role_index.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceRoleTest(unittest.TestCase):
    def test_attribution_partition_cannot_drop_or_duplicate_sources(self):
        valid = {role: [] for role in module.ROLES}
        valid.update(method=[0], background=[1], uncertain=[2])
        self.assertEqual(module.validate_roles(valid, 3), valid)
        for changed in (dict(valid, result=[0]), dict(valid, uncertain=[]),
                        dict(valid, uncertain=[3]), dict(valid, uncertain=[True])):
            with self.assertRaises(ValueError):
                module.validate_roles(changed, 3)
