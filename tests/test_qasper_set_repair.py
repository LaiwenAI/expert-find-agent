import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.qasper.set_repair import alternatives, gold_sets, outcome


def test_alternatives_keep_depth_and_enumerate_exact_swap_neighborhood():
    states, one_end = alternatives(16, 5)
    primary = set(range(5))
    assert states.shape == (606, 5)
    assert len({frozenset(row) for row in states}) == 606
    assert one_end == 56
    assert set(states[0]) == primary
    assert all(len(primary - set(row)) == 1 for row in states[1:one_end])
    assert all(len(primary - set(row)) == 2 for row in states[one_end:])


def test_alternative_gold_sets_apply_after_positions_are_remapped():
    paragraphs = list("abcdefg")
    ordering = [4, 3, 2, 1, 0, 5, 6]
    states, _ = alternatives(len(ordering), 5)
    target = gold_sets(paragraphs, ordering, states, [{"a", "f"}, {"b", "g"}])
    for chosen, label in zip(states, target, strict=True):
        evidence = {paragraphs[ordering[i]] for i in chosen}
        assert label == ({"a", "f"} <= evidence or {"b", "g"} <= evidence)


def test_repair_threshold_falls_back_on_equal_or_unset_gain():
    row = {"labels": np.array([True, False])}
    assert outcome(row, 1, 0.5, 0.5) == (True, False)
    assert outcome(row, 1, 0.5, None) == (True, False)
    assert outcome(row, 1, 0.5, 0.4) == (False, True)
