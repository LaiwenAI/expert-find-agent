import unittest
from expert_search.source_roles import ROLES, normalize_roles, prioritize_roles


class SourceRolesTest(unittest.TestCase):
    def test_conflicts_and_omissions_never_become_positive(self):
        value = {role: [] for role in ROLES}
        value.update(method=[0, 0, 1], background=[1])
        result = normalize_roles(value, 3)
        self.assertEqual(result['roles']['method'], [0])
        self.assertEqual(result['roles']['uncertain'], [1, 2])
        self.assertEqual((result['missing'], result['conflicts'], result['within_role_duplicates']), (1, 1, 1))
        self.assertEqual(prioritize_roles([2, 1, 0], result['roles']), [0, 2, 1])

    def test_foreign_ids_or_invalid_schema_rejected(self):
        for bad in ([3], [-1], [True], ['0']):
            value = {role: [] for role in ROLES}
            value['method'] = bad
            with self.assertRaises(ValueError):
                normalize_roles(value, 3)
        with self.assertRaises(ValueError):
            normalize_roles({'method': [0]}, 1)
