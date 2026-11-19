import pytest

pytest.importorskip("torch")

from experiments.decision import hotpot_set_repair_transfer as transfer


def case(identity, titles):
    return {"_id": identity, "question": f"Question {identity}",
            "context": [[title, ["", "identical sentence", "different continuation"]] for title in titles],
            "supporting_facts": [[title, 1] for title in titles]}


def test_sentence_identity_is_title_and_original_position():
    row = case("one", ["A", "B"])
    projected = transfer.project(row)
    assert projected[2] == ["identical sentence", "different continuation"] * 2
    assert projected[3] == ["A", "A", "B", "B"]
    assert projected[4] == {0, 2}
    row["supporting_facts"][0][1] = 0
    assert transfer.project(row) is None


def test_panel_split_drops_article_title_overlap(monkeypatch):
    monkeypatch.setattr(transfer, "SIZES", {"test": 1, "fit": 1})
    rows = [case("one", ["Common title", "B"]), case("two", ["COMMON TITLE", "C"]),
            case("three", ["D", "E"])]
    panels = transfer.split_hotpot(rows)
    assert panels["test"][0][0] != panels["fit"][0][0]
    assert any(p[0][0] == "three" for p in panels.values())
    assert transfer.split_hotpot(list(reversed(rows))) == panels
