import pytest

pytest.importorskip("torch")

from experiments.qasper.listwise_evidence import extract_window, stable_order


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_matched_window_keeps_the_same_budget_as_prefix():
    tokenizer = WordTokenizer()
    text = "one two three four five six seven calibration uncertainty ten"
    assert extract_window(tokenizer, text, 4) == "one two three four"
    window = extract_window(tokenizer, text, 4, {"calibration": 1., "uncertainty": 1.})
    assert window == "seven calibration uncertainty ten"
    assert len(window.split()) == 4


def test_candidate_position_order_is_repeatable_and_preserves_membership():
    ids = list(range(16))
    order = stable_order(ids, "question-A")
    assert sorted(order) == ids
    assert stable_order(list(reversed(ids)), "question-A") == order
    assert stable_order(ids, "question-B") != order
