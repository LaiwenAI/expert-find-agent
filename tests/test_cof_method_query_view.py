import pytest

pytest.importorskip("torch")

from experiments.retrieval.cof_method_query_view import build_view


def test_lowercase_abstract_supports_distinct_lead_and_method_views():
    query = (
        "review paper\nbackground opens. a challenge remains. few claims hold. "
        "we propose and train an algorithm. we benchmark and estimate."
    )
    assert build_view(query, "lead_sentences") == (
        "review paper\nbackground opens. a challenge remains. few claims hold."
    )
    assert build_view(query, "method_sentences") == (
        "review paper\nbackground opens. we propose and train an algorithm. we benchmark and estimate."
    )
    assert build_view(query, "method_instruction") == query
    assert build_view("title alone", "method_sentences") == "title alone"
