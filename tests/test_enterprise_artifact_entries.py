from experiments.retrieval.enterprise_artifact_entries import compile_entries


def fixture(rows):
    source = "\n".join(rows)
    start = 0
    windows = []
    for row in rows:
        windows.append(dict(expert="expert", view=dict(
            id="profile", start=start, end=start + len(row), text=row,
            roles=["profile_project_lead"],
        )))
        start += len(row) + 1
    return windows, {"profile": {"expert_id": "expert", "text": source}}


def test_specific_title_does_not_collapse_projects_with_shared_funder():
    rows = [
        "National Foundation, optical sensor calibration, project lead",
        "National Foundation, microwave sensor calibration, project lead",
        "National Foundation, optical sensor calibration, project lead",
    ]
    windows, sources = fixture(rows)
    first_column, _ = compile_entries(windows, sources)
    specific, audit = compile_entries(windows, sources, "specific")

    assert len(first_column) == 1
    assert [entry["view"]["text"] for entry in specific] == rows[:2]
    assert audit["duplicate_owner_entries"] == 1
    for entry in specific:
        view = entry["view"]
        assert sources[view["id"]]["text"][view["start"]:view["end"]] == view["text"]


def test_multiple_role_spans_make_one_line_ambiguous():
    line = "Optical sensor calibration, project lead; microwave sensor calibration, project lead"
    first, second = line.split("; ")
    windows, sources = fixture([line])
    first_view = dict(windows[0]["view"], end=len(first), text=first)
    second_view = dict(windows[0]["view"], start=len(first) + 2, text=second)

    entries, audit = compile_entries(
        [dict(expert="expert", view=first_view), dict(expert="expert", view=second_view)],
        sources, "specific",
    )

    assert not entries
    assert audit["ambiguous_or_long_lines"] == 1
