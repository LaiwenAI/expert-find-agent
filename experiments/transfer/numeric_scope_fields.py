"""Compare binary scope recognition with typed mismatch detection on bilingual evidence."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.numeric_evidence import constraint
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.bilingual_numeric_passages import Reader, compare_selected
from experiments.transfer.bilingual_numeric_scope import TASKS
from experiments.transfer.numeric_evidence_attribution import ATTRIBUTION, generate


PERSONAL = "Verify whether this excerpt explicitly says that the named candidate performed this test. " + ATTRIBUTION
MISMATCH = (
    "Check the requested subject, operating setting and metric independently against the reported test. "
    "A matching equipment name or metric cannot compensate for a different operating setting. "
    "Paraphrases of the same setting are acceptable. Ignore the numeric magnitude and the actor for this step. "
    "Return 0 for a measurement of the requested subject, metric and setting, "
    "1 if the setting differs or is missing, 2 if the subject or metric differs or is missing, "
    "or 3 if the excerpt only gives a target, estimate or proposed test. "
    "If more than one category applies, select any category other than 0."
)


class BatchedReader(Reader):
    def choose_batch(self, calls, classes):
        if not calls:
            return []
        messages = [
            [
                dict(role="system", content=instructions + " Return one digit."),
                dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ]
            for instructions, payload in calls
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, padding=True,
            padding_side="left", return_tensors="pt", return_dict=True,
        ).to("cuda")
        with self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits
            scores = logits[:, -1, self.labels[:classes]]
        return scores.argmax(dim=-1).tolist()


def verdict(reader, row, mode):
    bound = constraint(row["query"])
    if bound is None:
        raise ValueError("Expected one grounded bound")
    evidence, scope, candidate = row["evidence"], row["scope"], row["candidate"]
    start = perf_counter()
    scope_payloads = [dict(scope=scope, excerpt=excerpt) for excerpt in evidence]
    if mode == "typed_scope":
        scope_labels = reader.choose_batch([(MISMATCH, item) for item in scope_payloads], 4)
        scope_hits = [i for i, label in enumerate(scope_labels) if label == 0]
    elif mode == "batched_binary":
        prompt = TASKS["scope"] + " Use 0 for no, 1 for yes."
        scope_labels = reader.choose_batch([(prompt, item) for item in scope_payloads], 2)
        scope_hits = [i for i, label in enumerate(scope_labels) if label == 1]
    else:
        prompt = TASKS["scope"] + " Use 0 for no, 1 for yes."
        scope_labels = [reader.choose(prompt, item, (0, 1)) for item in scope_payloads]
        scope_hits = [i for i, label in enumerate(scope_labels) if label == 1]
    personal_prompt = PERSONAL + " Use 0 for no, 1 for yes."
    personal_payloads = [dict(candidate=candidate, excerpt=evidence[i]) for i in scope_hits]
    if mode == "sequential_binary":
        personal = [reader.choose(personal_prompt, item, (0, 1)) for item in personal_payloads]
    else:
        personal = reader.choose_batch([(personal_prompt, item) for item in personal_payloads], 2)
    picked = [i for i, label in zip(scope_hits, personal, strict=True) if label]
    outcome, _ = compare_selected(bound, evidence, picked)
    return dict(outcome=outcome, selected=picked, scope_labels=scope_labels, ms=(perf_counter() - start) * 1000)


MODES = ("sequential_binary", "batched_binary", "typed_scope")


def views(fixture, fresh):
    for cohort, path in (("development", fixture), ("new_scenarios", fresh)):
        specs = json.loads(path.read_bytes())
        aliases = {spec["id"]: spec.get("aliases", {}) for spec in specs}
        for row in generate(specs):
            replacements = aliases[row["id"]].get(row["language"], {})
            excerpts = row["evidence"]
            for original, replacement in replacements.items():
                excerpts = [excerpt.replace(original, replacement) for excerpt in excerpts]
            yield dict(row, cohort=cohort, evidence=excerpts)


def summarize(rows):
    report = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        report[cohort] = {}
        for language in ("en", "zh"):
            group = [row for row in rows if row["cohort"] == cohort and row["language"] == language]
            report[cohort][language] = {}
            for mode in MODES:
                counts = Counter(cases=len(group))
                errors = Counter()
                for row in group:
                    guess = row["methods"][mode]
                    expected = row["expected"]
                    counts["correct"] += guess["outcome"] == expected
                    counts["true_approvals"] += guess["outcome"] == expected == "eligible"
                    counts["exact_source_set"] += guess["selected"] == list(row["licensed"])
                    if guess["outcome"] == "eligible" and expected != "eligible":
                        counts["false_approvals"] += 1
                        errors[row["view"]] += 1
                report[cohort][language][mode] = dict(
                    counts, false_approvals=counts["false_approvals"], false_approval_views=dict(errors),
                    p95_ms=float(np.quantile([row["methods"][mode]["ms"] for row in group], .95)),
                )
    return report


def run(fixture, fresh, checkpoint, output):
    import torch

    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    source = list(views(fixture, fresh))
    ids = [(r["id"], r["language"], r["view"], r["order"]) for r in source]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate test view")
    for row in source:
        oracle, _ = compare_selected(constraint(row["query"]), row["evidence"], row["licensed"])
        if oracle != row["expected"]:
            raise ValueError(f"Invalid comparison after rewriting {row['id']}")
    reader = BatchedReader(checkpoint)
    for mode in MODES:
        verdict(reader, source[0], mode)
    rows = []
    for offset, row in enumerate(source):
        rotation = offset % len(MODES)
        modes = MODES[rotation:] + MODES[:rotation]
        rows.append(dict(row, methods={mode: verdict(reader, row, mode) for mode in modes}))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=reader.provenance,
        fixture_sha256=digest(fixture), fresh_sha256=digest(fresh), script_sha256=digest(Path(__file__)),
        report=summarize(rows), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "fresh", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
