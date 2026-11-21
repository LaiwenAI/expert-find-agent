"""Read all qualification clauses with shared forward and reverse context."""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_cross_precision import CONTROLS, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_qualification_student import target_index
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.enterprise_scope_transfer import classifier, make_dataset, summary

ARMS = ("per_clause", "shared_forward", "shared_symmetric")


def clauses_text(clauses, reverse):
    ordered = clauses[::-1] if reverse else clauses
    ends, cursor = [], 0
    for clause in ordered:
        cursor += len(clause)
        ends.append(cursor)
        cursor += 1
    return "\n".join(ordered), ends


class ScopeFeatures:
    def __init__(self, encoder):
        self.encoder = encoder

    def one(self, text, clauses, selected, arm):
        if arm == "per_clause":
            inputs = [
                json.dumps(dict(before=text[:clause["start"]], target=text[clause["start"]:clause["end"]],
                                after=text[clause["end"]:]), ensure_ascii=False)
                for clause in (clauses[i] for i in selected)
            ]
            lengths = self.encoder.tokenizer(inputs, add_special_tokens=True, return_length=True)["length"]
            return self.encoder.encode(inputs), max(lengths) <= self.encoder.max_length, sum(lengths)

        fragments = [text[clause["start"]:clause["end"]] for clause in clauses]
        formatted = [clauses_text(fragments, reverse) for reverse in range(1 + (arm == "shared_symmetric"))]
        encoded = self.encoder.tokenizer([item[0] for item in formatted], padding=True,
                                         return_offsets_mapping=True, return_tensors="pt")
        mappings = encoded.pop("offset_mapping").tolist()
        token_count = int(encoded.attention_mask.sum())
        if encoded.input_ids.shape[1] > 4096:
            raise ValueError("Request exceeds the experimental encoding limit")
        positions = []
        for direction, ((_, ends), mapping) in enumerate(zip(formatted, mappings, strict=True)):
            tokens = [(i, left) for i, (left, right) in enumerate(mapping) if right > left]
            starts = [start for _, start in tokens]
            cursor = [tokens[bisect_left(starts, end) - 1][0] for end in ends]
            if direction:
                cursor.reverse()
            positions.append([cursor[i] for i in selected])
        batch = encoded.to("cuda")
        states = self.encoder.model(**batch, use_cache=False).last_hidden_state
        anchors = normalize(torch.stack([row[index] for row, index in
                                         zip(states, positions, strict=True)]).float(), dim=-1)
        pooled = normalize(states[:, -1].float(), dim=-1)
        joined = torch.cat((anchors.mean(0), pooled.mean(0).expand(len(selected), -1)), dim=-1)
        return normalize(joined, dim=-1).cpu().numpy(), True, token_count


def gold(row):
    return row["role"] == "prior_qualification"


def features(reader, dataset, arm):
    result, eligible = [], []
    for row in dataset:
        clauses, index = target_index(row)
        values, covered, _ = reader.one(row["text"], clauses, [index], arm)
        result.extend(values)
        eligible.append(covered)
    return np.array(result), np.array(eligible)


def calculate(inputs, head):
    weight, center, intercept = head
    return (inputs - center) @ weight + intercept


def assess(dataset, scores, available, cutoff):
    expected = np.array([gold(row) for row in dataset])
    selected = (np.abs(scores) > cutoff) & available
    return summary(dataset, scores > 0) | dict(
        calibration_cutoff=cutoff, accepted=int(selected.sum()),
        accepted_errors=int(np.count_nonzero(selected & ((scores > 0) != expected))),
        accepted_qualified=int(np.count_nonzero(selected & expected)), within_encoding_limit=int(available.sum()),
    )


def timed(operation):
    torch.cuda.synchronize()
    start = perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, (perf_counter() - start) * 1000


def p50_p95(values):
    return np.quantile(values, [.50, .95]).tolist()


@torch.inference_mode()
def run(args):
    require_cuda(args.output)
    if digest(args.development) != DEVELOPMENT or digest(args.controls) != CONTROLS:
        raise ValueError("Expected frozen controls and development requests")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    shared = ScopeFeatures(encoder)
    parser = ComparisonParser(args.parser)
    for reader in parser.readers.values():
        reader.broadcast = True

    dataset = make_dataset()
    if dataset != json.loads(args.scope_dataset.read_bytes()):
        raise ValueError("Changed scope transfer controls")
    separate = json.loads(args.controls.read_bytes())
    for item in separate:
        item["group"] = item["pair"]
    dataset["heading"] = separate
    labels = np.array([gold(row) for row in dataset["train"]])
    calibration = np.array([gold(row) for row in dataset["calibration"]])
    heads, cutoffs, reports = {}, {}, {}
    for arm in ARMS:
        train, available = features(shared, dataset["train"], arm)
        if not available.all():
            raise ValueError("Training rows must fit")
        heads[arm] = classifier(train, labels, [row["group"] for row in dataset["train"]], False)
        x, available = features(shared, dataset["calibration"], arm)
        scores = calculate(x, heads[arm])
        errors = np.abs(scores[available & ((scores > 0) != calibration)])
        cutoffs[arm] = float(errors.max(initial=0))
        reports[arm] = {}
        for split in ("calibration", "held", "heading"):
            x, available = features(shared, dataset[split], arm)
            reports[arm][split] = assess(dataset[split], calculate(x, heads[arm]), available, cutoffs[arm])

    development = rows(args.development)
    if len(development) != 80 or any(row["split"] != "development" for row in development):
        raise ValueError("Expected 80 development rows")
    counters = {arm: Counter() for arm in ARMS}
    timings = {arm: [] for arm in (*ARMS, "reference")}
    guarded = {arm: [] for arm in ARMS}
    if args.reverse:
        development.reverse()
    comparators = (*ARMS, "reference")
    for i, row in enumerate(development):
        text = row["text"]
        found = spans(text)
        selected = list(range(len(found)))
        order = comparators[::-1] if args.reverse else comparators
        offset = i % len(order)
        outcomes, elapsed = {}, {}
        for arm in order[offset:] + order[:offset]:
            if arm == "reference":
                actual, elapsed[arm] = timed(lambda: evaluate(text, found, parser.readers, "consensus")["labels"])
                continue

            def predict():
                x, available, tokens = shared.one(text, found, selected, arm)
                scores = calculate(x, heads[arm])
                return scores > 0, available and bool((np.abs(scores) > cutoffs[arm]).all()), tokens

            outcomes[arm], elapsed[arm] = timed(predict)
        expected = actual == QUALIFICATION
        for arm, (prediction, accept, tokens) in outcomes.items():
            counters[arm].update(requests=1, clauses=len(found), tokens=tokens, skip_requests=int(accept),
                                 changed_requests=int(np.any(prediction != expected)),
                                 changed_skipped_requests=int(accept and np.any(prediction != expected)),
                                 model_misses=int((expected & ~prediction).sum()),
                                 model_additions=int((~expected & prediction).sum()))
            guarded[arm].append(elapsed[arm] + (0 if accept else elapsed["reference"]))
        for arm in comparators:
            timings[arm].append(elapsed[arm])

    write(args.output, dict(
        development={arm: dict(counters[arm], p50_p95_ms=p50_p95(timings[arm]),
                               simulated_guard_p50_p95_ms=p50_p95(guarded[arm])) for arm in ARMS},
        reference_p50_p95_ms=p50_p95(timings["reference"]),
        public=reports, reverse=args.reverse, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        embedding=verify_checkpoint(args.embedding)["revision"], reader=parser.readers["bf16"].provenance["revision"],
        inputs={key: digest(getattr(args, key)) for key in ("development", "controls", "scope_dataset")},
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "scope_dataset", "embedding", "parser", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
