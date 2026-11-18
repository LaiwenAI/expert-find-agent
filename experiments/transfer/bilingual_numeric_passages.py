"""Select comparable sentences and resolve numerical evidence with abstention."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.numeric_evidence import constraint, evaluate_measurement
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.bilingual_numeric_scope import TASKS, format_views


VIEWS = {
    "proposed_target": ("violates", "target_only"),
    "different_setting": ("meets", "wrong_setting"),
    "different_metric": ("violates", "wrong_metric"),
    "conflicting_results": ("meets", "violates"),
    "no_measurement": ("target_only", "wrong_setting"),
    "converted_units": ("converted", "wrong_metric"),
    "agreeing_results": ("meets", "converted"),
}
METHODS = ("direct", "joint_pointer", "independent_scope", "agreement", "oracle")
DECISIONS = ("reject", "eligible", "unknown")
LABELS = ((), (0,), (1,), (0, 1))


def cases(spec):
    singles = {(row["language"], row["view"]): row for row in format_views(spec)}
    for language in ("en", "zh"):
        for view, sequence in VIEWS.items():
            for order, names in enumerate((sequence, sequence[::-1])):
                pair = [singles[language, name] for name in names]
                evidence = [row["evidence"] for row in pair]
                licensed = tuple(i for i, row in enumerate(pair) if row["comparable"])
                votes = {pair[i]["expected"] for i in licensed}
                expected = votes.pop() if len(votes) == 1 else "unknown"
                yield dict(
                    id=spec["id"], language=language, view=view, order=order,
                    question=pair[0]["request"], scope=pair[0]["scope"],
                    evidence=evidence, licensed=licensed, expected=expected,
                )


def compare_selected(bound, sentences, selected):
    comparisons = [evaluate_measurement(bound, sentences[i]) for i in selected]
    labels = {label for label, _ in comparisons}
    if len(labels) != 1 or "unknown" in labels:
        return "unknown", []
    return labels.pop(), [quote for _, quote in comparisons]


class Reader:
    def __init__(self, checkpoint):
        import torch
        from experiments.qasper.paired_reader import load_reader

        self.torch = torch
        self.tokenizer, self.model, self.provenance = load_reader(checkpoint)
        coded = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(4)]
        if any(len(token) != 1 for token in coded):
            raise ValueError("Expected four single-token labels")
        self.labels = [token[0] for token in coded]

    def choose(self, instructions, content, alphabet):
        messages = [
            dict(role="system", content=instructions + " Return one digit."),
            dict(role="user", content=json.dumps(content, ensure_ascii=False)),
        ]
        request = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        with self.torch.inference_mode():
            scores = self.model(**request, use_cache=False, logits_to_keep=1).logits[0, -1, self.labels[:len(alphabet)]]
        return alphabet[int(scores.argmax().item())]


def evaluate(reader, record):
    bound = constraint(record["question"])
    if bound is None:
        raise ValueError("Expected one grounded bound")
    evidence, scope = record["evidence"], record["scope"]
    outcomes, selected, elapsed = {}, {}, {}

    start = perf_counter()
    outcomes["direct"] = reader.choose(
        TASKS["direct"] + " If comparable measured results give conflicting answers, return unknown. "
        "Use 0 for a verified violation, 1 for eligibility and 2 for unknown.",
        dict(question=record["question"], evidence=evidence), DECISIONS,
    )
    elapsed["direct"] = (perf_counter() - start) * 1000

    start = perf_counter()
    selected["joint_pointer"] = reader.choose(
        TASKS["scope"] + " Assess the two sentences separately. Choose 0 for neither, "
        "1 for the first only, 2 for the second only, or 3 for both.",
        dict(scope=scope, first=evidence[0], second=evidence[1]), LABELS,
    )
    outcomes["joint_pointer"], _ = compare_selected(bound, evidence, selected["joint_pointer"])
    elapsed["joint_pointer"] = (perf_counter() - start) * 1000

    start = perf_counter()
    selected["independent_scope"] = tuple(
        i for i, sentence in enumerate(evidence)
        if reader.choose(TASKS["scope"] + " Choose 0 for no, 1 for yes.",
                         dict(scope=scope, evidence=sentence), (False, True))
    )
    outcomes["independent_scope"], _ = compare_selected(bound, evidence, selected["independent_scope"])
    elapsed["independent_scope"] = (perf_counter() - start) * 1000

    selected["agreement"] = selected["joint_pointer"] if selected["joint_pointer"] == selected["independent_scope"] else ()
    outcomes["agreement"], _ = compare_selected(bound, evidence, selected["agreement"])
    elapsed["agreement"] = elapsed["joint_pointer"] + elapsed["independent_scope"]
    selected["oracle"] = record["licensed"]
    outcomes["oracle"], _ = compare_selected(bound, evidence, selected["oracle"])
    if outcomes["oracle"] != record["expected"]:
        raise ValueError("Oracle arithmetic disagrees with fixture")
    return dict(**record, outcomes=outcomes, selected=selected, ms=elapsed)


def summarize(rows):
    report = {}
    for language in ("en", "zh"):
        report[language] = {}
        for subset in ("all", *VIEWS):
            cohort = [r for r in rows if r["language"] == language and (subset == "all" or r["view"] == subset)]
            groups = {}
            for r in cohort:
                groups.setdefault((r["id"], r["view"]), []).append(r)
            report[language][subset] = {}
            for method in METHODS:
                metrics = Counter()
                for row in cohort:
                    guess = row["outcomes"][method]
                    metrics["cases"] += 1
                    metrics["correct"] += guess == row["expected"]
                    metrics["true_approvals"] += guess == "eligible" and row["expected"] == "eligible"
                    metrics["false_approvals"] += guess == "eligible" and row["expected"] != "eligible"
                    metrics["expected_approvals"] += row["expected"] == "eligible"
                    if method in row["selected"]:
                        metrics["exact_sentence_set"] += row["selected"][method] == row["licensed"]
                metrics["both_orders_correct"] = sum(
                    len(v) == 2 and all(r["outcomes"][method] == r["expected"] for r in v)
                    for v in groups.values()
                )
                if method != "oracle":
                    metrics["p95_ms"] = float(np.quantile([r["ms"][method] for r in cohort], .95))
                report[language][subset][method] = dict(metrics)
    return report


def run(fixture, checkpoint, output):
    import torch

    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    reader = Reader(checkpoint)
    base = [row for spec in json.loads(fixture.read_bytes()) for row in cases(spec)]
    if len(base) != 168 or len({(r["id"], r["language"], r["view"], r["order"]) for r in base}) != len(base):
        raise ValueError("Unexpected bilingual fixture shape")
    evaluate(reader, base[0])
    rows = [evaluate(reader, row) for row in base]
    write_json(output, dict(
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        model=reader.provenance, job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), rows=rows, report=summarize(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.fixture, args.checkpoint, args.output)
