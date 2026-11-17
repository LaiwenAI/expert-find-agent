import unittest

from experiments.bilingual.audit_parent_reuse import compact,has_shared_run


class ParentReuseAuditTest(unittest.TestCase):
    def test_literal_runs_ignore_case_and_whitespace_but_not_paraphrases(self):
        source=compact('Source Material with ROBOT CONTROL')
        self.assertTrue(has_shared_run(source,compact('robot control'),12))
        self.assertFalse(has_shared_run(source,compact('robot movement'),12))
        self.assertFalse(has_shared_run(source,compact('robot'),6))


if __name__=='__main__':unittest.main()
