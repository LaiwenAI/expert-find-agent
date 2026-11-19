from hashlib import sha256
import unittest

from experiments.retrieval.build_exact_enterprise_review import packet_rows


class ExactEnterpriseReviewTest(unittest.TestCase):
    def test_citation_set_sharing_preserves_candidate_and_blinding(self):
        queries = {"q": {"text": "One request"}}
        experts = {"e": {"name": "Casey", "institution": "Example University", "college": "Engineering"}}

        def view(identifier, text):
            return dict(id=identifier, start=0, end=len(text), sha256=sha256(text.encode()).hexdigest(),
                        text=text, kind="paper")

        first, second = view("p", "complete"), view("q", "another")
        methods = {
            "first": {"q": [("e", [first, second])]},
            "second": {"q": [("e", [second, first])]},
            "third": {"q": [("e", [view("p", "comp"), second])]},
        }
        forms, mapping, predictions, counts = packet_rows(queries, experts, methods)
        self.assertEqual(len(forms), 2)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(counts["shared_exact_views"], 1)
        self.assertEqual(counts["same_expert_distinct_views"], 1)
        case = lambda method: predictions[method][0]["recommendations"][0]["case_id"]
        self.assertEqual(case("first"), case("second"))
        self.assertNotEqual(case("first"), case("third"))
        self.assertEqual(forms[0]["candidate"], experts["e"])
        self.assertNotIn("method", forms[0])
