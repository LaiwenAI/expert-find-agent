from collections import Counter

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.qasper.evidence_gap_audit import diagnosis, expand


def test_diagnosis_accounts_for_alternative_gold_sets_and_output_capacity():
    paragraphs = list("abcdef")
    assert diagnosis(paragraphs, [0, 1], [0], [{"a"} | set("bcdef"), {"a"}]) == "complete"
    assert diagnosis(paragraphs, [0, 1], [0], [{"a", "b"}]) == "selection_miss"
    assert diagnosis(paragraphs, [0, 1], [0], [{"c"}]) == "shortlist_miss"
    assert diagnosis(paragraphs, [0, 1], [0], [set(paragraphs)]) == "output_capacity"
    assert diagnosis(paragraphs, [0], [0], []) == "unscored"


def test_gap_expansion_prefers_missing_term_and_same_section_neighbor():
    dense = np.linspace(0.9, 0.4, 10)
    sections = ["method"] * 9 + ["results"]
    counts = [Counter(["transformer"])] + [Counter() for _ in range(8)] + [Counter(["calibration"])]
    frequency = Counter(word for paragraph in counts for word in paragraph)
    selected = expand(dense, list(range(10)), [0], sections, (counts, frequency),
                      "transformer calibration", [0])
    assert selected[:2] == [9, 1]
    assert len(selected) == len(set(selected)) == 8
    assert 0 not in selected


def test_multi_panel_excludes_questions_with_an_alternative_single_source():
    from experiments.qasper.evidence_gap_audit import choose

    def qa(identity, sets):
        return {"question_id": identity, "answers": [
            {"answer": {"evidence": evidence, "unanswerable": False}} for evidence in sets
        ]}

    paper = {"full_text": [{"section_name": "Method", "paragraphs": ["a", "b"]}],
             "qas": [qa("alternative", [["a", "b"], ["a"]]), qa("requires_both", [["a", "b"]])]}
    selected = choose({"paper": paper}, multi=True)
    assert [item["question_id"] for _, item in selected] == ["requires_both"]
