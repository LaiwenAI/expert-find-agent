"""Stress numerical agreement of independent full-context layouts on public controls."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.retrieval.enterprise_cross_precision import (
    CONTROLS, NARROW, QUALIFICATION, WIDE, disagreements, prefix_cache, qualification_gap,
)
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import blocks, shared_prompt, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_recheck_layouts import ARMS, LayoutParser
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write

ENGLISH = (
    "Archival note. The following is a historical equipment management record, supplied as background. "
    "It is unrelated to the present procurement or to the applicants.",
    "Records staff logged dates for equipment delivery and handover. Copies were filed, "
    "and the electronic directory was used to locate them.",
    "The equipment went into storage after routine maintenance last year. The record lists "
    "the responsible unit, the storage rack and the time of handover.",
    "Auditors noted that the retired project used a separate archive identifier. "
    "Values in those archived attachments were retained for historical reference.",
    "The project coordinator inventoried older document versions. The office placed the "
    "files in a cabinet and kept them outside the present request attachments.",
    "The old ledger contained meeting minutes and a stock count. The meeting schedule "
    "was unrelated to the technical work in the current request.",
    "The current technical request begins in the following separate section.",
)
CHINESE = ["\u6863\u6848\u8bf4\u660e\u3002\u4ee5\u4e0b\u4ec5\u8bb0\u5f55\u5df2\u7ed3\u9879\u7684\u8bbe\u5907\u7ba1\u7406\u7ecf\u5386\uff0c\u4f9b\u9605\u8bfb\u80cc\u666f\u4f7f\u7528\u3002\u65e7\u6863\u6848\u4e0e\u672c\u6b21\u91c7\u8d2d\u9879\u76ee\u53ca\u7533\u8bf7\u4eba\u5458\u65e0\u5173\u3002","\u6863\u6848\u5458\u8bb0\u5f55\u4e86\u8bbe\u5907\u5165\u5e93\u548c\u79fb\u4ea4\u7684\u65e5\u671f\uff0c\u7eb8\u8d28\u8d44\u6599\u5df2\u5f52\u6863\uff0c\u7535\u5b50\u76ee\u5f55\u4f5c\u4e3a\u67e5\u627e\u7d22\u5f15\u3002","\u8bbe\u5907\u5728\u53bb\u5e74\u7684\u4f8b\u884c\u7ef4\u62a4\u540e\u8f6c\u5165\u4ed3\u5e93\u3002\u767b\u8bb0\u8868\u8bb0\u8f7d\u4e86\u7ecf\u529e\u5355\u4f4d\u3001\u4ed3\u5e93\u8d27\u67b6\u548c\u79fb\u4ea4\u65f6\u95f4\u3002","\u5ba1\u8ba1\u8bb0\u5f55\u6307\u51fa\uff0c\u65e7\u9879\u76ee\u4f7f\u7528\u5355\u72ec\u7684\u6863\u6848\u7f16\u53f7\uff0c\u9644\u4ef6\u4e2d\u7684\u5404\u9879\u6570\u636e\u4ec5\u7528\u4e8e\u5386\u53f2\u67e5\u8be2\u3002","\u9879\u76ee\u534f\u8c03\u5458\u6574\u7406\u4e86\u4ee5\u524d\u7684\u7248\u672c\u3002\u79d8\u4e66\u5904\u5c06\u6587\u4ef6\u5b58\u653e\u5728\u6863\u6848\u67dc\uff0c\u540c\u6279\u8d44\u6599\u6ca1\u6709\u4f5c\u4e3a\u9644\u4ef6\u968f\u9700\u6c42\u53d1\u5e03\u3002","\u65e7\u53f0\u8d26\u5305\u542b\u4f1a\u8bae\u7eaa\u8981\u53ca\u4ed3\u5e93\u6e05\u70b9\u62a5\u544a\u3002\u6b64\u524d\u7684\u4f1a\u8bae\u5b89\u6392\u4e0e\u672c\u6b21\u9700\u6c42\u4e2d\u7684\u6280\u672f\u4efb\u52a1\u65e0\u5173\u3002","\u4ee5\u4e0b\u53e6\u8d77\u4e00\u8282\uff0c\u5355\u72ec\u63cf\u8ff0\u672c\u6b21\u6280\u672f\u9700\u6c42\u3002"]


def sample_controls(rows):
    strata = defaultdict(list)
    for row in rows:
        strata[row["language"], row["layout"], row["role"]].append(row)
    return [row for key in sorted(strata) for row in sorted(
        strata[key], key=lambda x: sha256(x["id"].encode()).digest())[:2]]


def extended_text(row, repeats):
    header, *body, transition = CHINESE if row["language"] == "zh" else ENGLISH
    return "\n".join((header, *(body * repeats), transition, row["text"]))


def needs_recheck(parser, front, suffixes):
    bf16, fp16 = (parser.readers[x] for x in ("bf16", "fp16"))
    first, _ = bf16.cached(front, suffixes, prefix_cache(bf16, front))
    if qualification_gap(first) > WIDE:
        return False
    second, _ = fp16.cached(front, suffixes, prefix_cache(fp16, front))
    return disagreements(first, second) or qualification_gap(second) <= NARROW


def run(controls, models, output, reverse, length_matched):
    require_cuda(output)
    if digest(controls) != CONTROLS:
        raise ValueError("Expected the pinned bilingual synthetic controls")
    torch.backends.cuda.matmul.allow_tf32 = False
    parser = LayoutParser(ComparisonParser(models / "qwen3-4b-instruct-2507"))
    reader = parser.readers["bf16"]
    for active in parser.readers.values():
        active.broadcast = True
    selected = sample_controls(json.loads(controls.read_bytes()))
    observations = defaultdict(list)
    reference_shapes = defaultdict(list)
    modes = ARMS[::-1] if reverse else ARMS
    repeats_by_test = (8,) if length_matched else (1, 3)
    with torch.inference_mode():
        for repeats in repeats_by_test:
            for case, row in enumerate(selected):
                text = extended_text(row, repeats)
                clauses = spans(text)
                index, gold = target(text, clauses, row)
                front, suffixes = shared_prompt(reader.tokenizer, text, clauses)
                selected_block = next(block for block in blocks(front, suffixes) if index in dict(block))
                coordinates, decision_inputs = zip(*selected_block, strict=True)
                target_row = coordinates.index(index)
                trigger = needs_recheck(parser, front, decision_inputs) if length_matched else None
                reference_shapes[repeats].append([len(front), len(decision_inputs)])
                scores, work = {}, {}
                order = modes[case%len(modes):]+modes[:case%len(modes)]
                def forward(mode):
                    parser.mode = mode
                    parser.run_counts = Counter()
                    return (reader.independent(front, decision_inputs) if mode == ARMS[0]
                            else parser.recheck(reader, front, decision_inputs, None, None))[0]

                if case == 0:
                    for mode in order:
                        forward(mode).argmax(dim=1).cpu()
                for mode in order:
                    start = perf_counter()
                    logits = forward(mode)
                    scores[mode] = logits.argmax(dim=1).cpu().tolist()
                    work[mode] = dict(parser.run_counts, elapsed_ms=(perf_counter()-start)*1000)
                observations[repeats].append(dict(scores=scores, work=work, gold=gold, target_row=target_row, trigger=trigger))
    results = {}
    for repeats, rows in observations.items():
        results[repeats] = {
            "blocks": len(rows), "prefix_token_range": [min(x[0] for x in reference_shapes[repeats]),
                                                       max(x[0] for x in reference_shapes[repeats])],
            "natural_target_block_triggers": sum(bool(row["trigger"]) for row in rows) if length_matched else None,
            "arms": {mode: dict(
                changed_label_blocks=sum(row["scores"][mode]!=row["scores"][ARMS[0]] for row in rows),
                changed_qualification_blocks=sum(
                    [v == QUALIFICATION for v in row["scores"][mode]] !=
                    [v == QUALIFICATION for v in row["scores"][ARMS[0]]] for row in rows),
                target_correct=sum(row["gold"]==row["scores"][mode][row["target_row"]] for row in rows),
                changed_qualification_on_triggered_blocks=sum(
                    bool(row["trigger"]) and [v == QUALIFICATION for v in row["scores"][mode]] !=
                    [v == QUALIFICATION for v in row["scores"][ARMS[0]]] for row in rows) if length_matched else None,
                p50_p95_ms=np.quantile([row["work"][mode]["elapsed_ms"] for row in rows], [.5,.95]).tolist(),
                canonical_after_agreement=sum(row["work"][mode].get("agreement_rejects",0) for row in rows),
            ) for mode in ARMS},
        }
    write(output, dict(results=results, template_controls_sha256=CONTROLS, job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), reverse=reverse, length_matched=length_matched,
        runner_sha256=digest(Path(__file__)),
        recheck_runner_sha256=digest(Path(__file__).with_name("enterprise_recheck_layouts.py")),
        score_signature=sha256(json.dumps(dict(observations=[
            [{"scores": row["scores"], "trigger": row["trigger"]} for row in observations[repeats]]
            for repeats in repeats_by_test]), sort_keys=True).encode()).hexdigest()))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("controls","models","output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    cli.add_argument("--length-matched", action="store_true")
    run(**vars(cli.parse_args()))
