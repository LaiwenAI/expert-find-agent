from itertools import product

from experiments.retrieval.score_enterprise_evidence_gate import LEVEL, hit_bounds, paired_bounds


def test_paired_bounds_match_exhaustive_missing_label_assignments():
    frame = [{"id": "q", "group": "g"}, {"id": "r", "group": "g"}]
    alternatives = list(product((False, True), repeat=3))
    for labels in product(LEVEL.values(), repeat=3):
        known = dict(zip("abc", labels, strict=True))
        unknown = [key for key, level in known.items() if level == LEVEL["unresolved"]]
        for first_flags, second_flags in product(alternatives, repeat=2):
            def arm(flags):
                return {
                    "q": {key: known[key] for key, included in zip("ab", flags[:2], strict=True) if included},
                    "r": {"c": known["c"]} if flags[2] else {},
                }

            first, second = arm(first_flags), arm(second_flags)
            actual = paired_bounds(frame, first, second, top_k=2)
            hits, risks = [], []
            for replacements in product((LEVEL["adverse"], LEVEL["supported"]), repeat=len(unknown)):
                completed = known | dict(zip(unknown, replacements, strict=True))
                a, b = ([completed[key] for row in route.values() for key in row] for route in (first, second))
                hits.append((a.count(LEVEL["supported"]) - b.count(LEVEL["supported"])) / 4)
                risks.append(int(LEVEL["adverse"] in a) - int(LEVEL["adverse"] in b))
            assert actual["hit_delta"] == {"lower": min(hits), "upper": max(hits)}
            assert actual["adverse_group_rate_delta"] == {"lower": min(risks), "upper": max(risks)}


def test_hit_bounds_charge_empty_positions_and_failed_queries():
    views = {
        "returned": {"match": LEVEL["supported"], "unknown": LEVEL["unresolved"]},
        "failed": {},
    }
    assert hit_bounds(views, top_k=5) == {
        "confirmed": 1, "unresolved": 1, "slots": 10, "lower": 0.1, "upper": 0.2,
    }
