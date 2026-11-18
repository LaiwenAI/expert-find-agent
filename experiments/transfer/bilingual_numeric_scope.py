"""Separate numerical comparison from scope recognition in bilingual evidence checks."""

import argparse
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path

from expert_search.numeric_evidence import constraint, evaluate_measurement
from expert_search.qwen_reranker import QwenReranker, pack_tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


TASKS = {
    "direct": "Decide whether the cited evidence explicitly establishes the requested condition. "
              "Require a measured result for the exact metric in the stated setting meeting the numerical limit. "
              "A proposed target or a different operating setting cannot establish eligibility.",
    "scope": "Decide whether the cited evidence reports an actual measurement of the named metric "
             "for the same equipment and operating setting. A proposed target or a different metric "
             "or operating setting cannot establish a measurement. Do not compare numerical magnitudes.",
}
OPS = {"<=": "at most", "<": "strictly below", ">=": "at least", ">": "strictly above"}
ZH_OPS = {"<=": "\u4e0d\u8d85\u8fc7", "<": "\u4f4e\u4e8e",
          ">=": "\u81f3\u5c11", ">": "\u9ad8\u4e8e"}
THRESHOLDS = (.50, .90)
BATCH = 8
VIEWS = ("meets", "boundary", "violates", "converted", "target_only", "wrong_metric", "wrong_setting")


def format_views(spec):
    limit, op, unit = spec["limit"], spec["operator"], spec["unit"]
    direction = -1 if op.startswith("<") else 1
    for language, info in spec["text"].items():
        subject, setting, metric = (info[field] for field in ("subject", "setting", "metric"))
        local_unit = {"ms": "\u6beb\u79d2", "%": "\uff05"}[unit] if language == "zh" else unit
        if language == "en":
            scope = f"{subject}, {setting}, metric: {metric}."
            request = f"{scope} The requirement is {OPS[op]} {limit} {local_unit}."
        else:
            scope = f"{subject}\uff0c{setting}\uff0c\u6307\u6807\uff1a{metric}\u3002"
            request = f"{scope}\u8981\u6c42{ZH_OPS[op]} {limit} {local_unit}\u3002"
        for view in VIEWS:
            value = limit + 2 * direction
            actual_setting, actual_metric, rendered_unit = setting, metric, local_unit
            measured = view != "target_only"
            if view == "boundary":
                value = limit
            elif view == "violates":
                value = limit - 2 * direction
            elif view == "converted":
                if unit == "ms":
                    value = Decimal(value) / 1000
                    rendered_unit = "\u79d2" if language == "zh" else "s"
                elif language == "en":
                    rendered_unit = "percent"
            elif view == "wrong_metric":
                actual_metric = info["other_metric"]
            elif view == "wrong_setting":
                actual_setting = info["other_setting"]
            comparable = measured and actual_metric == metric and actual_setting == setting
            if language == "en":
                verb = "The measured result was" if measured else "The proposed target was"
                source = f"For the {subject} {actual_setting}, {verb} {value} {rendered_unit} for {actual_metric}."
            else:
                verb = "\u5b9e\u6d4b\u7ed3\u679c\u4e3a" if measured else "\u62df\u5b9a\u76ee\u6807\u4e3a"
                source = f"{subject}\uff0c{actual_setting}\uff0c{actual_metric}{verb} {value} {rendered_unit}\u3002"
            approved = comparable and (view != "violates" and (view != "boundary" or "=" in op))
            expected = "eligible" if approved else "reject" if comparable else "unknown"
            yield dict(id=spec["id"], view=view, language=language, scope=scope, request=request,
                       evidence=source, comparable=comparable, expected=expected)


def score(model, rows, mode):
    jobs = []
    for i, row in enumerate(rows):
        query = row["request"] if mode == "direct" else row["scope"]
        text = f"<Instruct>: {TASKS[mode]}\n<Query>: {query}\n<Document>: {row['evidence']}"
        body = model.tokenizer.encode(text, add_special_tokens=False)
        encoded, total = pack_tokens(model.prefix, body, model.suffix, model.max_length)
        if len(encoded) != total:
            raise ValueError("Unexpected truncation")
        jobs.append((i, encoded))
    jobs.sort(key=lambda item: len(item[1]))
    for start in range(0, len(jobs), BATCH):
        batch = jobs[start:start + BATCH]
        margins = model._batch([tokens for _, tokens in batch])
        for (i, _), value in zip(batch, margins, strict=True):
            if not 0 <= value <= 1:
                raise ValueError("Nonfinite or invalid model score")
            rows[i]["scope_score" if mode == "scope" else "direct"] = value


def summarize(rows):
    report = {}
    for language in ("en", "zh"):
        cohort = [row for row in rows if row["language"] == language]
        report[language] = {}
        for arm in cohort[0]["verdicts"]:
            counts = Counter()
            views = {}
            for view in VIEWS:
                subset = [row for row in cohort if row["view"] == view]
                approved = [row["verdicts"][arm] == "eligible" for row in subset]
                positives = [row["expected"] == "eligible" for row in subset]
                metrics = dict(
                    cases=len(subset), expected_approvals=sum(positives),
                    true_approvals=sum(p and y for p, y in zip(approved, positives, strict=True)),
                    false_approvals=sum(p and not y for p, y in zip(approved, positives, strict=True)),
                )
                counts.update(metrics)
                views[view] = metrics
            report[language][arm] = dict(totals=counts, views=views)
    return report


def score_reader(rows, checkpoint):
    import torch

    from experiments.qasper.paired_reader import load_reader

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    tokenizer, model, provenance = load_reader(checkpoint)
    coded = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(3)]
    if any(len(ids) != 1 for ids in coded):
        raise ValueError("Expected one token per label")
    labels = [ids[0] for ids in coded]
    orders = {"direct": ("reject", "eligible", "unknown"), "scope": ("unknown", "comparable")}
    roles = {
        "direct": " Reply 0 if a comparable measured result violates the numerical limit, 1 if it meets the limit, "
                  "or 2 if no comparable measured result is stated. Return only one digit.",
        "scope": " Reply 0 if no corresponding empirical measurement exists or 1 if one exists. Return only one digit.",
    }
    with torch.inference_mode():
        for i, row in enumerate(rows):
            modes = tuple(TASKS)
            for mode in modes[::1 if i % 2 == 0 else -1]:
                text = json.dumps(dict(
                    query=row["request"] if mode == "direct" else row["scope"], evidence=row["evidence"]
                ), ensure_ascii=False)
                request = tokenizer.apply_chat_template(
                    [dict(role="system", content=TASKS[mode] + roles[mode]), dict(role="user", content=text)],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                ).to("cuda")
                logits = model(**request, use_cache=False, logits_to_keep=1).logits[0, -1, labels[:len(orders[mode])]]
                row[mode + "_decision"] = orders[mode][int(logits.argmax().item())]
    return provenance, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name())


def run(fixture, checkpoint, output, scope_reader=False):
    if output.exists():
        raise FileExistsError(output)
    specs = json.loads(fixture.read_bytes())
    rows = [row for spec in specs for row in format_views(spec)]
    runtime = {}
    if scope_reader:
        provenance, runtime = score_reader(rows, checkpoint)
    else:
        provenance = verify_checkpoint(checkpoint)
        model = QwenReranker(checkpoint, max_length=384)
        for mode in TASKS:
            score(model, rows, mode)
    for row in rows:
        bound = constraint(row["request"])
        if bound is None:
            raise ValueError("Failed to parse a fixture bound")
        numeric, quote = evaluate_measurement(bound, row["evidence"])
        if quote is None or quote not in row["evidence"]:
            raise ValueError("Expected a single grounded quantity")
        verdicts = {"oracle_scope": numeric if row["comparable"] else "unknown"}
        if verdicts["oracle_scope"] != row["expected"]:
            raise ValueError("Fixture semantics and numeric oracle disagree")
        if scope_reader:
            verdicts["direct_reader"] = row["direct_decision"]
            verdicts["factored_reader"] = numeric if row["scope_decision"] == "comparable" else "unknown"
        else:
            for threshold in THRESHOLDS:
                verdicts[f"direct_{threshold:.2f}"] = "eligible" if row["direct"] >= threshold else "unknown"
                verdicts[f"scope_{threshold:.2f}"] = numeric if row["scope_score"] >= threshold else "unknown"
        row["verdicts"] = verdicts
    write_json(output, dict(fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
                            checkpoint=provenance, runtime=runtime, rows=rows, report=summarize(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--scope-reader", action="store_true")
    args = parser.parse_args()
    run(args.fixture, args.checkpoint, args.output, args.scope_reader)
