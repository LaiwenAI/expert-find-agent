"""Time source-prefix reuse against separate and batched expert checks."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval
from experiments.qasper.paired_reader import load_reader
from experiments.qasper.adaptive_evidence import digest


ARMS = ("serial", "batch", "shared")
REPEATS = 3
SAMPLE = 48
LABELS = ("A", "B", "C")


def cases(retrieval):
    _, papers, reviewers, _ = load("SciRepEval")
    raw = retrieval / "SciRepEval.json"
    manifest = json.loads((retrieval / "summary.json").read_bytes())
    if digest(raw) != manifest["datasets"]["SciRepEval"]["rows_sha256"]:
        raise ValueError("CoF retrieval ID mismatch")
    rows = json.loads(raw.read_bytes())
    output = []
    for row in rows:
        appearances = defaultdict(list)
        for expert in row["arms"]["full"]["hits"]:
            for witness in expert["evidence"]:
                appearances[witness["id"]].append(expert["id"])
        shared = [(doc, eids) for doc, eids in appearances.items() if len(set(eids)) > 1]
        if not shared:
            continue
        doc, experts = min(shared, key=lambda pair: (-len(set(pair[1])), pair[0]))
        qid = row["query_id"]
        q, source = papers[qid], papers[doc]
        prefix = ("Determine whether the shared source supports this candidate's expertise to review "
                  "the submission. Use candidate-specific evidence.\n\nSubmission:\n"
                  + q["title"] + "\n" + q["abstract"] + "\n\nShared publication:\n"
                  + source["title"] + "\n" + source["abstract"] + "\n\n")
        suffixes = []
        for expert in sorted(set(experts)):
            owned = set(reviewers[expert]["papers"])
            if doc not in owned:
                raise ValueError("Invalid paper ownership")
            other = next(hit for hit in row["arms"]["full"]["hits"] if hit["id"] == expert)
            titles = [papers[w["id"]]["title"] for w in other["evidence"] if w["id"] != doc]
            if any(w["id"] not in owned for w in other["evidence"]):
                raise ValueError("Invalid auxiliary ownership")
            suffixes.append("Other documented publications of the candidate:\n" + "\n".join(titles)
                            + "\n\nA: supports. B: does not support. C: insufficient. Answer:")
        output.append((qid, doc, prefix, suffixes))
    return sorted(output, key=lambda row: sha256(row[0].encode()).digest())[:SAMPLE]


def prepare(tokenizer, prefix, suffixes):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    labels = [encode(label) for label in LABELS]
    if any(len(label) != 1 for label in labels):
        raise ValueError("Require one token per choice")
    front = encode(prefix)
    ends = [encode(suffix) for suffix in suffixes]
    if not front or any(not value for value in ends):
        raise ValueError("Empty verifier input")
    sequences = [front + end for end in ends]
    width = max(map(len, sequences))
    padding = [width - len(s) for s in sequences]

    def tensor(value):
        return torch.tensor(value, dtype=torch.long, device="cuda")

    padded = tensor([[tokenizer.pad_token_id] * sizes + s for sizes, s in zip(padding, sequences)])
    mask = tensor([[0] * sizes + [1] * len(s) for sizes, s in zip(padding, sequences)])
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0)
    return dict(prefix=tensor([front]), suffixes=[tensor([end]) for end in ends],
                serial=[tensor([s]) for s in sequences], batch=padded, mask=mask, positions=positions,
                label_ids=tensor([label[0] for label in labels]))


def measure(reader, inputs, method):
    label_ids = inputs["label_ids"]
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        if method == "serial":
            logits = [reader(input_ids=seq, logits_to_keep=1, use_cache=False).logits[0, -1, label_ids]
                      for seq in inputs["serial"]]
            scores = torch.stack(logits)
        elif method == "batch":
            output = reader(input_ids=inputs["batch"], attention_mask=inputs["mask"],
                            position_ids=inputs["positions"], logits_to_keep=1, use_cache=False)
            scores = output.logits[:, -1, label_ids]
        else:
            prefix = inputs["prefix"]
            prompt_length = prefix.shape[1]
            cache = reader(input_ids=prefix, use_cache=True, logits_to_keep=1).past_key_values
            logits = []
            for end in inputs["suffixes"]:
                if cache.get_seq_length() != prompt_length:
                    raise ValueError("Unexpected verifier cache length")
                output = reader(input_ids=end, past_key_values=cache, logits_to_keep=1, use_cache=True)
                logits.append(output.logits[0, -1, label_ids])
                cache.crop(-end.shape[1])
            scores = torch.stack(logits)
        result = scores.float().cpu().numpy()
    return result, (perf_counter() - start) * 1000


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    instances = cases(args.retrieval)
    if len(instances) != SAMPLE:
        raise ValueError("Unexpected shared source group count")
    tokenizer, reader, checkpoint = load_reader(args.model)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = []
    for i, (query, source, prefix, suffixes) in enumerate(instances):
        inputs = prepare(tokenizer, prefix, suffixes)
        if max(seq.shape[1] for seq in inputs["serial"]) > reader.config.max_position_embeddings:
            raise ValueError("Verifier context exceeds window")
        timings, outputs = defaultdict(list), {}
        if i == 0:
            for arm in ARMS:
                measure(reader, inputs, arm)
        for repeat in range(REPEATS):
            offset = (i + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                scores, elapsed = measure(reader, inputs, arm)
                if arm in outputs:
                    np.testing.assert_array_equal(scores, outputs[arm])
                outputs[arm] = scores
                timings[arm].append(elapsed)
        reference = outputs["serial"]
        rows.append(dict(query_id=query, source_id=source, experts=len(suffixes),
                         prefix_tokens=inputs["prefix"].shape[1], timings_ms=timings,
                         match={arm: int(np.sum(scores.argmax(axis=1) == reference.argmax(axis=1)))
                                for arm, scores in outputs.items()},
                         max_absolute_logit_error={arm: float(np.abs(scores-reference).max())
                                                   for arm, scores in outputs.items()}))
    report = {}
    for arm in ARMS:
        # Median across the three repeats is the per-request sample.
        latency = [float(np.median(row["timings_ms"][arm])) for row in rows]
        report[arm] = {
            "mean_group_ms": mean(latency),
            "group_p50_p95_ms": list(map(float, np.quantile(latency, [.50, .95]))),
            "choices_matching_serial": sum(row["match"][arm] for row in rows),
            "max_absolute_logit_error": max(row["max_absolute_logit_error"][arm] for row in rows),
        }
    paired = {arm: paired_interval([
        float(np.median(row["timings_ms"]["shared"]) - np.median(row["timings_ms"][arm]))
        for row in rows]) for arm in ("serial", "batch")}
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "groups.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    summary = dict(checkpoint=checkpoint, gpu=torch.cuda.get_device_name(), groups=len(rows),
                   expert_checks=sum(row["experts"] for row in rows),
                   prefix_tokens_mean=mean(row["prefix_tokens"] for row in rows), repeats=REPEATS,
                   metrics=report, shared_minus=paired, groups_sha256=digest(output),
                   source_rows_sha256=digest(args.retrieval / "SciRepEval.json"),
                   scope="Shared-source groups only. Scores are mechanically compared to separate views; "
                         "no author-level evidence labels. Timing includes model forward and reading scores "
                         "to host; excludes tokenization, retrieval, model initialization and final generation.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
