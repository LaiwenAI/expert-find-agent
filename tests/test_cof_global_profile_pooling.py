import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.retrieval.cof_global_profile_pooling import centers


def test_profile_modes_preserve_distinct_topics():
    rows = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float32)
    centroid, modes = centers(rows)
    assert np.allclose(centroid, [1 / np.sqrt(2)] * 2)
    assert {tuple(row) for row in modes} == {(1., 0.), (0., 1.)}
    assert np.allclose(np.linalg.norm(modes, axis=1), 1.)


def test_one_publication_cannot_create_an_extra_mode():
    centroid, modes = centers(np.array([[.6, .8]], dtype=np.float32))
    assert np.allclose(modes, [centroid, centroid])
