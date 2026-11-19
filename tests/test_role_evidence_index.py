import unittest

from expert_search.role_evidence_index import RoleEvidenceIndex


class RoleEvidenceIndexTest(unittest.TestCase):
    def test_preserves_earlier_opaque_witness_and_verifies_it_once(self):
        index = RoleEvidenceIndex([
            ("first", [("early", None), ("later", {"design", "analysis"})]),
            ("second", [("next", {"design", "analysis"})]),
        ])
        calls = []

        def read(owner, source):
            calls.append((owner, source))
            return {"design", "analysis"}

        result = index.rank(("design", "analysis"), read, top=1)
        self.assertEqual(result["ranking"], [("first", ("early", "early"))])
        self.assertEqual(calls, [("first", "early")])

    def test_excludes_only_complete_negative_candidates(self):
        index = RoleEvidenceIndex([
            ("excluded", [("design", {"design"}), ("negative", set())]),
            ("unavailable", [("gone", None)]),
            ("qualified", [("design", {"design"}), ("analysis", {"analysis"})]),
        ])
        requests = ("design", "analysis")
        read = lambda *_: None
        posted = index.rank(requests, read, top=1, postings=True)
        forward = index.rank(requests, read, top=1, postings=False)
        automatic = index.rank(requests, read, top=1)
        self.assertEqual(posted["ranking"], [("qualified", ("design", "analysis"))])
        self.assertEqual(automatic, forward)
        self.assertEqual(posted["ranking"], forward["ranking"])
        self.assertEqual(posted["reads"], forward["reads"])
        self.assertEqual(posted["visited_candidates"], forward["visited_candidates"] - 1)
