"""Recheck ambiguous cached clause blocks using the original independent batch."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_isolated_parsing import IsolatedReader, ROLES, blocks, shared_prompt, spans


ARMS = ("independent", "cache", "margin_recheck", "hash_recheck")
CUTOFFS = (.125, .25, .5, 1., 2., 4.)
SALT = "clause-fixed-recheck-v1"


def choose_cutoff(controls):
    """Choose the smallest validation band attaining maximum block recovery."""
    def trial(cutoff):
        wrong = sum(row["wrong"] for row in controls if row["margin"] > cutoff)
        checks = sum(row["margin"] <= cutoff for row in controls)
        return dict(cutoff=cutoff, wrong=wrong, rechecked_blocks=checks, blocks=len(controls))

    trials = [trial(cutoff) for cutoff in (-1., *CUTOFFS)]
    best = min(trials, key=lambda row: (row["wrong"], row["rechecked_blocks"], row["cutoff"]))
    return best, trials


def hash_recheck(request, block, rate):
    key = sha256(f"{SALT}:{request}:{block}".encode()).digest()[:8]
    return int.from_bytes(key, "big") < rate * 2**64


class GuardedReader(IsolatedReader):
    @torch.inference_mode()
    def evaluate(self, text, clauses, method, cutoff=-1., rate=0.):
        start = perf_counter()
        front, suffixes = shared_prompt(self.tokenizer, text, clauses)
        blocks_seen = rechecked = slots = 0
        traces = []
        coordinates, all_labels = [], []
        prefix_cache = None
        if method != "independent":
            prefix = torch.tensor([front], device="cuda")
            prefix_cache = self.backbone(input_ids=prefix, use_cache=True).past_key_values
            if prefix_cache.get_seq_length() != len(front):
                raise RuntimeError("Incomplete context cache")
            slots += len(front)
        identity = sha256(text.encode()).hexdigest() if method == "hash_recheck" else None
        for block in blocks(front, suffixes):
            indices, suffix_batch = zip(*block, strict=True)
            margin = None
            if method == "independent":
                logits, size = self.independent(front, suffix_batch)
            else:
                logits, size = self.cached(front, suffix_batch, prefix_cache)
                if method in ("margin_recheck", "calibration"):
                    values = logits.float().topk(2, dim=1).values
                    margin = float((values[:, 0] - values[:, 1]).min().item())
                selected = ((method == "margin_recheck" and margin <= cutoff)
                            or (method == "hash_recheck" and hash_recheck(identity, blocks_seen, rate)))
                if selected:
                    logits, checked_size = self.independent(front, suffix_batch)
                    size += checked_size
                    rechecked += 1
            coordinates.extend(indices)
            all_labels.append(logits.argmax(dim=1))
            if method == "calibration":
                traces.append((indices, margin))
            slots += size
            blocks_seen += 1
        labels = np.empty(len(clauses), dtype=np.int64)
        labels[coordinates] = torch.cat(all_labels).cpu().numpy()
        return dict(labels=labels, blocks=blocks_seen, rechecked=rechecked, input_slots=slots,
                    ms=(perf_counter() - start) * 1000, traces=traces)


def target(text, clauses, item):
    if "target" not in item:
        return None, None
    matches = [i for i, clause in enumerate(clauses) if text[clause["start"]:clause["end"]] == item["target"]]
    if len(matches) != 1:
        raise ValueError("Ambiguous anchored target")
    return matches[0], ROLES.index(item["role"])


def summarize(rows):
    report = {}
    for method in ARMS:
        tally = Counter()
        times = []
        for row in rows:
            current = row["arms"][method]
            reference = row["arms"]["independent"]["labels"]
            labels = current["labels"]
            tally["clauses"] += len(labels)
            tally["label_disagreements"] += int((labels != reference).sum())
            tally["requests_with_disagreement"] += int(not np.array_equal(labels, reference))
            tally["added_qualifications"] += int(((labels == 2) & (reference != 2)).sum())
            tally["removed_qualifications"] += int(((labels != 2) & (reference == 2)).sum())
            for key in ("blocks", "rechecked", "input_slots"):
                tally[key] += current[key]
            if row["target"] is not None:
                predicted = int(labels[row["target"]])
                tally["targets_correct"] += predicted == row["gold"]
                tally["false_qualifications_on_controls"] += predicted == 2 and row["gold"] != 2
                tally["missed_qualifications_on_controls"] += predicted != 2 and row["gold"] == 2
            times.append(current["ms"])
        report[method] = dict(tally) | {"p50_p95_ms": np.quantile(times, [.5, .95]).tolist()}
    return dict(requests=len(rows), arms=report)


def run(development, controls, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a new output and Slurm CUDA")
    public = json.loads(controls.read_bytes())
    real = [json.loads(line) for line in development.read_bytes().splitlines()]
    if len(real) != 80 or any(row["split"] != "development" for row in real):
        raise ValueError("Use only the pinned development export")
    reader = GuardedReader(checkpoint)
    warm = spans(public[0]["text"])
    for method in ARMS:
        reader.evaluate(public[0]["text"], warm, method)
    calibration = []
    for item in public:
        text = item["text"]
        clauses = spans(text)
        reference = reader.evaluate(text, clauses, "independent")["labels"]
        cached = reader.evaluate(text, clauses, "calibration")
        for indices, margin in cached["traces"]:
            keys = list(indices)
            calibration.append(dict(margin=margin, wrong=int((reference[keys] != cached["labels"][keys]).sum())))
    best, grid = choose_cutoff(calibration)
    rate = best["rechecked_blocks"] / best["blocks"]
    print("Frozen on public controls:", best, flush=True)
    reports = {}
    for name, cohort in (("controls", public), ("development", real)):
        rows = []
        for i, item in enumerate(cohort):
            text = item["text"]
            clauses = spans(text)
            idx, gold = target(text, clauses, item)
            offset = i % len(ARMS)
            arms = {method: reader.evaluate(text, clauses, method, best["cutoff"], rate)
                    for method in ARMS[offset:] + ARMS[:offset]}
            rows.append(dict(arms=arms, target=idx, gold=gold))
            if (i + 1) % 20 == 0:
                print(name, i + 1, "of", len(cohort), flush=True)
        reports[name] = summarize(rows)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=reader.provenance,
        controls_sha256=digest(controls), development_sha256=digest(development),
        script_sha256=digest(Path(__file__)), base_reader_sha256=digest(Path(__file__).with_name("enterprise_isolated_parsing.py")),
        selected=best, grid=grid, hash_reference_rate=rate, report=reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
