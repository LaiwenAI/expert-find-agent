"""Audit incrementally published enterprise demands before evaluating search."""

import argparse
from collections import Counter
from pathlib import Path
import json

from openpyxl import load_workbook

from experiments.enterprise_review_gate import design, scenario_power
from experiments.external_enterprise_frame import (
    BATCHES, COUNTS, digest_file, historical, nantong, package, row,
)
from experiments.freeze_enterprise_panel import issuer_key, screen
from experiments.retrieval.enterprise_gate_design_audit import paired_pass_probability, passing_boundary


WORKBOOKS = {
    "wuhu": (147, "\u9700\u6c42\u540d\u79f0", "\u9700\u6c42\u7b80\u4ecb"),
    "weifang": (79, "\u6280\u672f\u96be\u9898\u548c\u9700\u6c42\u540d\u79f0",
                "\u6280\u672f\u96be\u9898\u548c\u9700\u6c42\u7684\u4e3b\u8981\u5185\u5bb9\u548c\u6280\u672f\u6307\u6807"),
}
ISSUER_FIELDS = {"\u4f01\u4e1a\u540d\u79f0", "\u5355\u4f4d\u540d\u79f0", "\u53d1\u699c\u5355\u4f4d"}
SEQUENCE = "\u5e8f\u53f7"


def workbook_profile(path, schema):
    expected, title, body = schema
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if len(book.worksheets) != 1:
            raise ValueError("Expected one published demand worksheet")
        cells = iter(book.active.values)
        next(cells)
        header = next(cells)
        sequence, name, content = [header.index(field) for field in (SEQUENCE, title, body)]
        if any(value in ISSUER_FIELDS for value in header):
            raise ValueError("Issuer field appeared; audit the new publication schema")
        listed, complete = [], 0
        for values in cells:
            value = values[sequence]
            if isinstance(value, int) or (isinstance(value, str) and value.strip().isdecimal()):
                listed.append(int(value))
                complete += bool(str(values[name] or "").strip() and str(values[content] or "").strip())
        if sorted(listed) != list(range(1, expected + 1)):
            raise ValueError("Publication rows differ from the expected inventory")
        return dict(listed=len(listed), complete_title_and_body=complete, issuer_field_present=False,
                    admitted_to_issuer_frame=0, attachment_sha256=digest_file(path))
    finally:
        book.close()


def usable(rows):
    return [r for r in rows if r["complete"] and issuer_key(r["company"])]


def freeze_snapshot(earlier):
    records, files = [], {}
    for batch, count in zip(BATCHES, COUNTS, strict=True):
        batch_rows, files[batch] = package(earlier, batch, count)
        records.extend(batch_rows)
    path = earlier / "nantong.html"
    records.extend(nantong(path))
    files["nantong"] = dict(notice_sha256=digest_file(path))
    return records, files


def run(earlier, snapshot, workbook, protocol, previous, output, private_index):
    if output.exists() or private_index.exists():
        raise FileExistsError("Require fresh output paths")
    older, earlier_hashes = freeze_snapshot(earlier)
    original = json.loads(previous.read_bytes())
    if original["sources"] != earlier_hashes:
        raise ValueError("Earlier public snapshot changed")
    first, first_hashes = package(snapshot, "first", 30)
    anonymous = {name: workbook_profile(snapshot / f"{name}.xlsx", schema) | {
        "notice_sha256": digest_file(snapshot / f"{name}.html")
    } for name, schema in WORKBOOKS.items()}
    old_usable, new_usable = usable(older), usable(first)
    history = historical(workbook)
    old_selected, _ = screen(old_usable + history, set(), set())
    if len(old_selected) != original["screen"]["selected_groups"]:
        raise ValueError("The original frame did not reproduce")
    after, details = screen(old_usable + new_usable + history, set(), set())
    separate, disjoint = screen(new_usable + [r | {"split": "prior"} for r in old_usable] + history, set(), set())
    sources = {r["id"]: r["batch"] for r in older + first}
    spec = json.loads(protocol.read_bytes())
    gates = [spec[k] for k in ("coverage_floor", "conditional_risk_ceiling", "confidence")]
    scenarios = []
    alpha = (1. - spec["confidence"]) / 2
    k = spec["top_k"]
    for name, n in (("published_baseline", len(old_selected)), ("expanded_candidates", len(after)),
                    ("planning_500", 500)):
        superiority = passing_boundary(n, k, spec["minimum_grounded_p5_gain"], alpha)
        noninferiority = passing_boundary(n, k, -spec["quality_noninferiority_margin"], alpha, inclusive=True)
        paired = [dict(endpoint=endpoint, gain=gain, discordance=discordance,
                       probability=paired_pass_probability(n, k, boundary, gain, discordance))
                  for endpoint, boundary, gain in (("superiority", superiority, .03),
                                                   ("noninferiority", noninferiority, 0.))
                  for discordance in (.20, .40)]
        scenarios.append(dict(name=name, groups=n, gate=design(n, *gates), paired_slots=paired,
                              power=[dict(coverage=.75, adverse=risk,
                                          probability=scenario_power(n, *gates, .75, risk))
                                     for risk in (.01, .02, .03)]))
    report = dict(
        prior_public_source_hashes=earlier_hashes, added_xian_source=first_hashes,
        xian_first=dict(listed=len(first), complete=sum(r["complete"] for r in first),
                        issuer_and_complete=len(new_usable), separate_after_exposure=len(separate)),
        anonymous_workbook_audits=anonymous, historical_records=len(history),
        earlier_provisional_groups=len(old_selected), expanded_provisional_groups=len(after),
        net_candidate_groups=len(after) - len(old_selected),
        selected_by_batch=dict(Counter(sources[r["id"]] for r in after)),
        combined_screen={k: v for k, v in details.items() if k != "original_sealed_groups"},
        new_source_exposure_screen={k: v for k, v in disjoint.items() if k != "original_sealed_groups"},
        planning_scenarios=scenarios, protocol_sha256=digest_file(protocol),
        script_sha256=digest_file(Path(__file__)), expert_queries_run=0, human_review_labels_added=0,
    )
    private = dict(historical_workbook_sha256=digest_file(workbook), records=after,
                   first_batch_disjoint=separate, expanded_source_hashes=earlier_hashes | {"first": first_hashes})
    with private_index.open("x", encoding="utf-8") as stream:
        private_index.chmod(0o600)
        json.dump(private, stream, ensure_ascii=True)
        stream.write("\n")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("earlier", "snapshot", "workbook", "protocol", "previous", "output", "private_index"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
