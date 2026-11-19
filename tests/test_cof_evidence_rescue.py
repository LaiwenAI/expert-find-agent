import pytest

pytest.importorskip("torch")

from experiments.retrieval.cof_evidence_rescue import budgeted_frontier


class Reader:
    def __init__(self, margins, rechecks=None):
        self.margins = margins
        self.rechecks = rechecks or {}
        self.steps = []

    def score(self, jobs):
        self.steps.extend(("score", doc) for _, doc in jobs)
        return [self.margins[doc] for _, doc in jobs]

    def recheck(self, jobs):
        self.steps.extend(("recheck", doc) for _, doc in jobs)
        return [self.rechecks[doc] for _, doc in jobs]


def test_budget_exhaustion_never_confirms_uncertain_source():
    pool = [("first", ("a", "b")), ("second", ("c", "d"))]
    reader = Reader({"a": .1, "b": .9, "c": -.9, "d": -.9}, {"a": -.9})
    ranking, spent, certified = budgeted_frontier(pool, reader, 4)
    assert ranking == [["first", []], ["second", []]]
    assert spent == 4 and not certified
    ranking, spent, certified = budgeted_frontier(pool, reader, 5)
    assert ranking == [["first", []], ["second", []]]
    assert spent == 5 and certified


def test_reused_source_is_charged_once_and_promotes_verified_expert():
    pool = [("first", ("a", "b")), ("second", ("b", "c"))]
    reader = Reader({"a": -.9, "b": .9, "c": .9})
    ranking, spent, certified = budgeted_frontier(pool, reader, 3)
    assert ranking == [["second", ["b", "c"]], ["first", []]]
    assert spent == 3 and certified
    assert reader.steps.count(("score", "b")) == 1
