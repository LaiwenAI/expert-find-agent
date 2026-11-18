"""Distinguish personally performed tests from reviews and other contributors' measurements."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.numeric_evidence import constraint
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.bilingual_numeric_passages import DECISIONS, Reader, compare_selected
from experiments.transfer.bilingual_numeric_scope import TASKS, format_views


VIEWS = {
    "direct_support": (("meets", "performs"), ("wrong_metric", "performs")),
    "review_only": (("meets", "reviews"), ("wrong_setting", "performs")),
    "other_person_passes": (("meets", "other"), ("violates", "performs")),
    "other_person_only": (("meets", "other"), ("target_only", "performs")),
    "personal_conflict": (("meets", "performs"), ("violates", "performs")),
    "other_person_conflicts": (("meets", "performs"), ("violates", "other")),
    "joint_work": (("converted", "joint"), ("wrong_metric", "other")),
    "proposed_only": (("target_only", "performs"), ("wrong_setting", "performs")),
}
METHODS = ("direct", "scope_only", "joint_check", "factor_fields", "oracle")
ACTORS = {"en": ("Analyst Lin", "Analyst Wu"), "zh": ("\u6797\u5de5", "\u5434\u5de5")}
ATTRIBUTION = (
    "The named candidate must personally perform or jointly perform the measurement in this evidence. "
    "Reading, summarizing, or reviewing someone else's test provides no proof of personal execution."
)


def render(language, actor, other, role, evidence, style):
    if language == "en":
        patterns = {
            "performs": (f"{actor} carried out the following test. ", f"The following trial was run by {actor}. "),
            "reviews": (f"{actor} examined the report from the test performed by {other}. ",
                        f"{other} ran the trial. {actor} only reviewed the report. "),
            "other": (f"{other} ran the trial. ", f"This measurement was performed by {other}. "),
            "joint": (f"{actor} and {other} ran the trial together. ",
                      f"Both {actor} and {other} jointly performed this test. "),
        }
    else:
        patterns = {
            "performs": (f"{actor}\u5b9e\u65bd\u4e86\u4ee5\u4e0b\u6d4b\u8bd5\u3002",
                         f"\u4ee5\u4e0b\u6d4b\u8bd5\u7531{actor}\u5b8c\u6210\u3002"),
            "reviews": (f"{actor}\u53ea\u5ba1\u9605\u62a5\u544a\uff0c\u5b9e\u9645\u6d4b\u8bd5\u7531{other}\u5b8c\u6210\u3002",
                        f"{other}\u5b8c\u6210\u6d4b\u8bd5\u3002{actor}\u53ea\u9605\u8bfb\u4e86\u62a5\u544a\u3002"),
            "other": (f"{other}\u5b9e\u65bd\u4e86\u4ee5\u4e0b\u6d4b\u8bd5\u3002",
                      f"\u4ee5\u4e0b\u5b9e\u6d4b\u7531{other}\u5b8c\u6210\u3002"),
            "joint": (f"{actor}\u4e0e{other}\u5171\u540c\u5b8c\u6210\u6d4b\u8bd5\u3002",
                      f"\u6d4b\u8bd5\u7531{actor}\u548c{other}\u4e00\u8d77\u5b9e\u65bd\u3002"),
        }
    return patterns[role][style] + evidence


def generate(specs):
    for index, spec in enumerate(specs):
        views = {(row["language"], row["view"]): row for row in format_views(spec)}
        for language in ACTORS:
            actor, other = ACTORS[language][::1 if index % 2 == 0 else -1]
            for variant, specs_for_view in VIEWS.items():
                for order, entries in enumerate((specs_for_view, specs_for_view[::-1])):
                    chosen = [views[language, name] for name, _ in entries]
                    sources = [render(language, actor, other, role, row["evidence"], index % 2)
                               for row, (_, role) in zip(chosen, entries, strict=True)]
                    comparable = tuple(i for i, row in enumerate(chosen) if row["comparable"])
                    credited = tuple(i for i, (_, role) in enumerate(entries) if role in {"performs", "joint"})
                    licensed = tuple(i for i in comparable if i in credited)
                    values = {chosen[i]["expected"] for i in licensed}
                    expected = values.pop() if len(values) == 1 else "unknown"
                    yield dict(id=spec["id"], language=language, view=variant, order=order,
                               candidate=actor, query=chosen[0]["request"], scope=chosen[0]["scope"],
                               evidence=sources, comparable=comparable, credited=credited,
                               licensed=licensed, expected=expected)


def evaluate(reader, row):
    bound = constraint(row["query"])
    if bound is None:
        raise ValueError("Expected a single bound")
    actor, evidence, scope = row["candidate"], row["evidence"], row["scope"]
    predictions, recognized, runtimes = {}, {}, {}

    def call(prompt, payload):
        return reader.choose(prompt + " Use 0 for no, 1 for yes.", payload, (False, True))

    start = perf_counter()
    predictions["direct"] = reader.choose(
        TASKS["direct"] + " " + ATTRIBUTION +
        " Treat unresolved or conflicting personally attributable readings as unknown. "
        "Use 0 for a verified violation, 1 for eligibility and 2 for unknown.",
        dict(candidate=actor, question=row["query"], excerpts=evidence), DECISIONS,
    )
    runtimes["direct"] = (perf_counter()-start)*1000
    start = perf_counter()
    scope_hits = [i for i, text in enumerate(evidence)
                  if call(TASKS["scope"], dict(scope=scope, excerpt=text))]
    recognized["scope_only"] = tuple(scope_hits)
    predictions["scope_only"], _ = compare_selected(bound, evidence, recognized["scope_only"])
    runtimes["scope_only"] = (perf_counter()-start)*1000
    start = perf_counter()
    joint = [i for i, text in enumerate(evidence) if call(
        TASKS["scope"] + " " + ATTRIBUTION,
        dict(candidate=actor, scope=scope, excerpt=text),
    )]
    recognized["joint_check"] = tuple(joint)
    predictions["joint_check"], _ = compare_selected(bound, evidence, recognized["joint_check"])
    runtimes["joint_check"] = (perf_counter()-start)*1000
    start = perf_counter()
    personal_hits = [i for i in scope_hits if call(
        "Verify whether this excerpt explicitly says that the named candidate performed this test. " + ATTRIBUTION,
        dict(candidate=actor, excerpt=evidence[i]),
    )]
    recognized["factor_fields"] = tuple(personal_hits)
    predictions["factor_fields"], _ = compare_selected(bound, evidence, recognized["factor_fields"])
    runtimes["factor_fields"] = runtimes["scope_only"] + (perf_counter()-start)*1000
    recognized["oracle"] = row["licensed"]
    predictions["oracle"], _ = compare_selected(bound, evidence, row["licensed"])
    if predictions["oracle"] != row["expected"]:
        raise ValueError("Oracle failed to reproduce constructed label")
    return dict(**row, predictions=predictions, recognized=recognized, ms=runtimes)


def summarize(rows):
    report = {}
    for language in ACTORS:
        report[language] = {}
        for view in ("all", *VIEWS):
            batch = [r for r in rows if r["language"] == language and (view == "all" or r["view"] == view)]
            report[language][view] = {}
            for method in METHODS:
                result = Counter()
                for row in batch:
                    expected, prediction = row["expected"], row["predictions"][method]
                    result["cases"] += 1
                    result["correct"] += expected == prediction
                    result["true_approvals"] += expected == prediction == "eligible"
                    result["false_approvals"] += expected != "eligible" and prediction == "eligible"
                    if method in row["recognized"]:
                        sources = row["recognized"][method]
                        result["exact_source_set"] += sources == row["licensed"]
                        result["wrong_attribution_selections"] += sum(i not in row["credited"] for i in sources)
                if method != "oracle":
                    result["p95_ms"] = float(np.quantile([r["ms"][method] for r in batch], .95))
                report[language][view][method] = dict(result)
    return report


def run(fixture, checkpoint, output):
    import torch

    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    source = list(generate(json.loads(fixture.read_bytes())))
    if len(source) != 192 or len({(r["id"], r["language"], r["view"], r["order"]) for r in source}) != len(source):
        raise ValueError("Unexpected bilingual case inventory")
    reader = Reader(checkpoint)
    evaluate(reader, source[0])
    rows = [evaluate(reader, row) for row in source]
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=reader.provenance,
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        rows=rows, report=summarize(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    run(args.fixture, args.checkpoint, args.output)
