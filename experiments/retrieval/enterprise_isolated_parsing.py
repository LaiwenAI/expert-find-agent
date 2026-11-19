"""Share the full requirement context across independently scored clause decisions."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import linear

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_requirement_roles import BATCH_ROWS, PROMPT, ROLES, ScopedReader, spans


ARMS = ("independent", "prefix_cache", "isolated_pack", "local")
MARKER = "ENTERPRISE_TARGET_CLAUSE_PLACEHOLDER"
MAX_BLOCK = 8192


def shared_prompt(tokenizer, text, clauses):
    block = json.dumps({"request": text}, ensure_ascii=False)
    messages = [
        dict(role="system", content=PROMPT),
        dict(role="user", content=f"{block}\nClassify this target within the request:\n{MARKER}"),
    ]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if template.count(MARKER) != 1:
        raise ValueError("Expected a unique target placeholder")
    left, right = template.split(MARKER)
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    front = encode(left)
    suffixes = [encode(json.dumps({
        "start": span["start"], "end": span["end"], "target": text[span["start"]:span["end"]],
    }, ensure_ascii=False) + right) for span in clauses]
    if not front or any(not suffix or len(front) + len(suffix) > MAX_BLOCK for suffix in suffixes):
        raise ValueError("A full clause decision exceeds the configured token budget")
    return front, suffixes


def blocks(front, suffixes):
    current, length = [], len(front)
    for i, suffix in sorted(enumerate(suffixes), key=lambda item: (len(item[1]), item[0])):
        if len(current) == BATCH_ROWS or length + len(suffix) > MAX_BLOCK:
            yield current
            current, length = [], len(front)
        current.append((i, suffix))
        length += len(suffix)
    if current:
        yield current


def padded(sequences, pad_id, offset=0):
    width = max(map(len, sequences))
    gaps = [width - len(sequence) for sequence in sequences]
    input_ids = torch.tensor([[pad_id] * gap + sequence for gap, sequence in zip(gaps, sequences, strict=True)],
                             device="cuda")
    mask = torch.arange(width, device="cuda")[None, :] >= torch.tensor(gaps, device="cuda")[:, None]
    return input_ids, mask, (mask.cumsum(1) - 1).clamp(min=0) + offset


class IsolatedReader(ScopedReader):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.backbone = self.model.model
        self.head = self.model.get_output_embeddings().weight[self.labels].contiguous()

    def independent(self, front, suffixes):
        ids, mask, positions = padded([front + suffix for suffix in suffixes], self.tokenizer.pad_token_id)
        hidden = self.backbone(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        return linear(hidden.last_hidden_state[:, -1].contiguous(), self.head), ids.numel()

    def cached(self, front, suffixes, prefix_cache):
        cache = deepcopy(prefix_cache)
        cache.batch_repeat_interleave(len(suffixes))
        ids, mask, positions = padded(suffixes, self.tokenizer.pad_token_id, len(front))
        prefix_mask = torch.ones((len(suffixes), len(front)), dtype=torch.bool, device="cuda")
        hidden = self.backbone(input_ids=ids, attention_mask=torch.cat((prefix_mask, mask), 1),
                               position_ids=positions, past_key_values=cache, use_cache=True)
        return linear(hidden.last_hidden_state[:, -1].contiguous(), self.head), ids.numel()

    def packed(self, front, suffixes):
        tokens = front + [token for suffix in suffixes for token in suffix]
        group_ids = [0] * len(front)
        positions = list(range(len(front)))
        ends = []
        for i, suffix in enumerate(suffixes, 1):
            group_ids.extend([i] * len(suffix))
            positions.extend(range(len(front), len(front) + len(suffix)))
            ends.append(len(positions) - 1)
        order = torch.arange(len(tokens), device="cuda")
        groups = torch.tensor(group_ids, device="cuda")
        allowed = (order[:, None] >= order[None, :]) & ((groups[:, None] == groups[None, :]) | (groups[None, :] == 0))
        mask = torch.zeros((len(tokens), len(tokens)), device="cuda", dtype=self.head.dtype)
        mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        hidden = self.backbone(input_ids=torch.tensor([tokens], device="cuda"), attention_mask=mask[None, None],
                               position_ids=torch.tensor([positions], device="cuda"), use_cache=False)
        return linear(hidden.last_hidden_state[0, ends], self.head), len(tokens)

    @torch.inference_mode()
    def classify(self, text, clauses, method):
        if method == "local":
            local, count = self.judge(text, clauses, range(len(clauses)), full=False)
            return np.array([local[i][0] for i in range(len(clauses))]), None, count
        front, suffixes = shared_prompt(self.tokenizer, text, clauses)
        prefix_cache = None
        tokens = 0
        if method == "prefix_cache":
            prefix = torch.tensor([front], device="cuda")
            prefix_cache = self.backbone(input_ids=prefix, use_cache=True).past_key_values
            if prefix_cache.get_seq_length() != len(front):
                raise RuntimeError("Incomplete prefix cache")
            tokens += len(front)
        result = torch.empty((len(clauses), len(ROLES)), dtype=torch.float32, device="cuda")
        for block in blocks(front, suffixes):
            indices, suffix_batch = zip(*block, strict=True)
            if method == "independent":
                scores, count = self.independent(front, suffix_batch)
            elif method == "prefix_cache":
                scores, count = self.cached(front, suffix_batch, prefix_cache)
            else:
                scores, count = self.packed(front, suffix_batch)
            result[list(indices)] = scores.float()
            tokens += count
        values = result.cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite classification scores")
        return values.argmax(axis=1), values, tokens


def evaluate(reader, row, index):
    text = row["text"]
    clauses = spans(text)
    if not clauses:
        raise ValueError("Expected at least one clause")
    target, gold = None, None
    if row.get("target"):
        matches = [i for i, span in enumerate(clauses) if text[span["start"]:span["end"]] == row["target"]]
        if len(matches) != 1:
            raise ValueError("Ambiguous controlled target")
        target, gold = matches[0], ROLES.index(row["role"])
    offset = index % len(ARMS)
    arms = {}
    for method in ARMS[offset:] + ARMS[:offset]:
        start = perf_counter()
        labels, values, tokens = reader.classify(text, clauses, method)
        arms[method] = dict(labels=labels, logits=values, tokens=tokens, ms=(perf_counter() - start) * 1000)
    return dict(arms=arms, pair=row.get("pair"), target=target, gold=gold)


def summarize(rows):
    report = {}
    for method in ARMS:
        results = [row["arms"][method] for row in rows]
        exact = Counter()
        errors = []
        pairs = defaultdict(list)
        for row, result in zip(rows, results, strict=True):
            reference = row["arms"]["independent"]
            choices = result["labels"]
            exact["spans"] += len(choices)
            exact["equal_labels"] += int((choices == reference["labels"]).sum())
            exact["equal_requests"] += bool(np.array_equal(choices, reference["labels"]))
            exact["qualification_disagreements"] += int(((choices == 2) != (reference["labels"] == 2)).sum())
            if result["logits"] is not None:
                errors.append(float(np.max(np.abs(result["logits"] - reference["logits"]))))
            if row["target"] is not None:
                predicted = int(choices[row["target"]])
                correct = predicted == row["gold"]
                exact["controls"] += 1
                exact["controls_correct"] += correct
                exact["false_qualifications"] += predicted == 2 and row["gold"] != 2
                exact["missed_qualifications"] += predicted != 2 and row["gold"] == 2
                pairs[row["pair"]].append(correct)
        report[method] = dict(
            **exact, both_pair_roles_correct=sum(len(pair) == 2 and all(pair) for pair in pairs.values()),
            max_abs_label_logit_drift=max(errors) if errors else None,
            forward_input_slots=sum(row["tokens"] for row in results),
            p50_p95_ms=np.quantile([row["ms"] for row in results], [.5, .95]).tolist(),
        )
    return dict(requests=len(rows), arms=report)


def run(development, controls, checkpoint, output, controls_only=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a fresh output and Slurm CUDA")
    controlled = json.loads(controls.read_bytes())
    dev = [json.loads(line) for line in development.read_bytes().splitlines()]
    if len(dev) != 80 or any(row["split"] != "development" for row in dev):
        raise ValueError("Use only the 80-request development export")
    reader = IsolatedReader(checkpoint)
    evaluate(reader, controlled[0], 0)
    reports = {}
    cohorts = [("controls", controlled)] if controls_only else [("controls", controlled), ("development", dev)]
    for name, cohort in cohorts:
        rows = []
        for i, row in enumerate(cohort):
            rows.append(evaluate(reader, row, i))
            if (i + 1) % 20 == 0:
                print(name, i + 1, "of", len(cohort), flush=True)
        reports[name] = summarize(rows)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=reader.provenance,
        script_sha256=digest(Path(__file__)), controls_sha256=digest(controls),
        development_sha256=digest(development), reader_sha256=digest(Path(__file__).with_name("enterprise_requirement_roles.py")),
        report=reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--controls-only", action="store_true")
    run(**vars(parser.parse_args()))
