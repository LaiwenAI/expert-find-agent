from itertools import product

import numpy as np
import pytest

from experiments.credit.source_role_offset_lattice import best_paths


def legal(path):
    collapsed = [label for i, label in enumerate(path) if label and (i == 0 or label != path[i - 1])]
    return len(collapsed) == len(set(collapsed)) and (
        3 not in collapsed or (1 in collapsed and collapsed.index(1) < collapsed.index(3))
    )


@pytest.mark.parametrize("length", [4, 5, 6])
def test_dynamic_program_matches_exhaustive_path_and_margins(length):
    logits = np.random.default_rng(length).normal(size=(length, 5))
    paths = [p for p in product(range(5), repeat=length) if legal(p)]

    def score(path):
        return sum(logits[i, label] - logits[i, 0] for i, label in enumerate(path))

    full = sorted((p for p in paths if {1, 2, 3, 4}.issubset(p)), key=score, reverse=True)
    partial = [p for p in paths if not {1, 2, 3, 4}.issubset(p)]
    best, margins = best_paths(logits)
    assert best == full[0]
    assert margins["alternative"] == pytest.approx(score(full[0]) - score(full[1]))
    assert margins["presence"] == pytest.approx(score(full[0]) - max(map(score, partial)))


def test_handles_ties_and_inputs_too_short_for_four_spans():
    assert best_paths(np.zeros((3, 5))) == ((), {})
    path, margins = best_paths(np.zeros((6, 5)))
    assert legal(path) and {1, 2, 3, 4}.issubset(path)
    assert margins == {"alternative": 0., "presence": 0.}
