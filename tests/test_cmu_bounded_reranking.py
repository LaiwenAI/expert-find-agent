import numpy as np

from experiments.decision.cmu_bounded_reranking import evidence_order


def test_both_bounded_routes_read_late_winner_and_skip_impossible_sources():
    dense = np.array([.81, .79, .78, .71, .68, .60, .54, .51, .41, .37, .31])
    margins = np.array([-9., -4., 12., 12., 12., 12., 12., 12., 12., 12., 12.])

    def scorer(left, right):
        return margins[left:right], 0

    full = evidence_order(dense, scorer, "full")
    assert full[0] == 2
    assert evidence_order(dense, scorer, "first_two")[0] == 0
    for arm in ("bounded", "bounded_upfront"):
        route = evidence_order(dense, scorer, arm)
        assert route[:2] == full[:2]
        assert 3 <= route[2] < full[2]


def test_scores_at_the_same_maximum_keep_earliest_dense_order():
    dense = np.array([.9, .9, .7])

    def scorer(left, right):
        return [100.] * (right - left), 0

    assert {evidence_order(dense, scorer, arm)[0]
            for arm in ("full", "bounded", "bounded_upfront")} == {0}
