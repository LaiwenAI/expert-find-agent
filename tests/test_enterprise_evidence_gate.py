import unittest

from experiments.retrieval.score_enterprise_evidence_gate import FIELDS, counts


def outcome(label, status):
    return {field: {"label": label, "status": status} for field in FIELDS}


class EnterpriseEvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(id="paper", start=0, end=4, sha256="a" * 64)
        self.frame = [dict(id=f"q{i}", group=group) for i, group in enumerate(("a", "a", "b", "c"))]
        self.mapping = {
            f"c{i}": dict(query_id=f"q{i}", expert_id=f"e{i}", sources=[self.source | {"alias": "first"}])
            for i in range(3)
        }
        self.outputs = [
            dict(query_id=f"q{i}", status="issued",
                 recommendations=[dict(case_id=f"c{i}", expert_id=f"e{i}", sources=[self.source])])
            for i in range(3)
        ] + [dict(query_id="q3", status="failed", recommendations=[])]
        self.outcomes = {
            "c0": outcome(None, "missing_review"),
            "c1": outcome("not_supported_by_view", "agreed"),
            "c2": outcome("uncertain", "adjudicated_uncertain"),
        }

    def tally(self):
        return counts(self.frame, self.outputs, self.mapping, self.outcomes, FIELDS, 5)

    def test_groups_count_once_and_adverse_dominates_unknown(self):
        result = self.tally()
        self.assertEqual((result["groups"], result["issued"], result["adverse"], result["unresolved"]), (3, 2, 1, 1))
        self.assertEqual(result["query_status"]["failed"], 1)

    def test_extra_citation_invalidates_a_view_even_for_same_expert(self):
        self.outputs[0]["recommendations"][0]["sources"].append(self.source | {"id": "extra"})
        with self.assertRaisesRegex(ValueError, "frozen reviewed evidence"):
            self.tally()

    def test_omitted_query_and_inconsistent_reconciliation_are_rejected(self):
        omitted = self.outputs.pop()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.tally()
        self.outputs.append(omitted)
        self.outcomes["c0"] = outcome("supported", "missing_review")
        with self.assertRaisesRegex(ValueError, "reconciliation status"):
            self.tally()
