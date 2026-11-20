"""Batch both role queries and answer encodings in one normalization request."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit.study_clause_selection import ROLE_BY_TASK, parse_clauses
from experiments.credit.study_role_normalization import SCHEMA, extended_ids, variant
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


SYSTEM = ("Map the requested duty to one contribution field. Use outside if the duty does not "
          "follow from any listed field, or if more than one field would be required. Reply with "
          "the digit of the best field.")
ARMS = ("reference", "serial_last", "two_batches", "one_batch")
WIDTH = dict(serial_last=1, two_batches=2, one_batch=4)


def menu_inputs(tasks):
    menus = (tuple(SCHEMA), tuple(reversed(SCHEMA)))
    return [(dict(duty=task, fields={str(i): SCHEMA[field] for i, field in enumerate(menu)}), menu)
            for task in tasks for menu in menus]


def batched(judge, inputs):
    torch = judge.torch
    sequences = []
    for view, _ in inputs:
        rendered = judge.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(view, ensure_ascii=False)}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        sequences.append(rendered["input_ids"][0])
    length = max(len(sequence) for sequence in sequences)
    token = judge.tokenizer.pad_token_id
    if token is None:
        token = judge.tokenizer.eos_token_id
    ids = torch.full((len(inputs), length), token, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, -len(sequence):] = sequence
        mask[i, -len(sequence):] = 1
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
    with torch.inference_mode():
        logits = judge.model(input_ids=ids.to("cuda"), attention_mask=mask.to("cuda"),
                             position_ids=positions.to("cuda"), use_cache=False,
                             logits_to_keep=1).logits[:, -1, judge.ids]
        choices = logits.argmax(dim=1).cpu().tolist()
    return [menu[index] for (_, menu), index in zip(inputs, choices, strict=True)]


def normalize(judge, tasks, arm):
    started = perf_counter()
    inputs = menu_inputs(tasks)
    if arm == "reference":
        predictions = [judge(SYSTEM, view, menu)[0] for view, menu in inputs]
    else:
        width = WIDTH[arm]
        predictions = [
            item for index in range(0, len(inputs), width)
            for item in batched(judge, inputs[index:index + width])
        ]
    agreed = tuple(a if a == b else "outside"
                   for a, b in zip(predictions[::2], predictions[1::2], strict=True))
    return predictions, agreed, (perf_counter() - started) * 1000


def data_views(data):
    for case in data["rows"]:
        roles = [ROLE_BY_TASK[value] for value in case["tasks"]]
        valid = [variant(case, role, i) for i, role in enumerate(roles)]
        yield case, valid, True
        yield case, [valid[0], variant(case, "outside", 2)], False


def evaluate(judge, data, order_offset):
    views = list(data_views(data))
    for arm in ARMS:
        normalize(judge, views[0][1], arm)
    timings = {arm: [] for arm in ARMS}
    parity = {arm: Counter() for arm in ARMS}
    outcomes = {arm: Counter() for arm in ARMS}
    for i, (case, tasks, supported_query) in enumerate(views):
        shift = (i + order_offset) % len(ARMS)
        estimates = {arm: normalize(judge, tasks, arm) for arm in ARMS[shift:] + ARMS[:shift]}
        reference, agreed_ref, _ = estimates["reference"]
        names = case["initials"]
        for arm, (choices, agreed, elapsed) in estimates.items():
            timings[arm].append(elapsed)
            parity[arm].update(
                menu_choices=len(choices),
                unequal_menu_choices=sum(a != b for a, b in zip(choices, reference, strict=True)),
                unequal_query_mappings=int(agreed != agreed_ref),
            )
            fields = set(agreed)
            correct = []
            for j, note in enumerate(case["notes"]):
                bound = {role for _, role, owners in parse_clauses(note) if names in owners}
                eligible = "outside" not in fields and len(fields) == len(agreed) and fields <= bound
                expected = supported_query and j == case["positive_index"]
                correct.append(eligible == expected)
                outcomes[arm].update(positive_accept=int(expected and eligible),
                                     in_schema_wrong_accept=int(supported_query and not expected and eligible),
                                     outside_wrong_accept=int(not supported_query and eligible))
            if supported_query:
                outcomes[arm].update(both_correct=int(all(correct)))
    return dict(queries=len(views), outcomes=outcomes, parity=parity, timing_ms={
        arm: dict(p50=float(np.quantile(values, .50)), p95=float(np.quantile(values, .95)))
        for arm, values in timings.items()
    })


def main():
    import torch

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    data = json.loads(args.fixture.read_bytes())
    judge = Judge()
    extended_ids(judge)
    result = evaluate(judge, data, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
        fixture_sha256=digest(args.fixture), model_provenance_sha256=judge.sha,
        **result,
    ))


if __name__ == "__main__":
    main()
