"""Compare whole-block and compact rechecks of cached requirement decisions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import IsolatedReader, ROLES, blocks, shared_prompt, spans


ARMS = ("independent", "cache", "block_recheck", "row_recheck", "coalesced_recheck")
CUTOFF = 2.


def recheck_batches(front, suffixes, schedule, selected, mode):
    if mode == "coalesced_recheck":
        indices = sorted(selected)
        for compact in blocks(front, [suffixes[i] for i in indices]):
            yield [(indices[i], suffix) for i, suffix in compact]
        return
    for block in schedule:
        ambiguous = [(i, suffix) for i, suffix in block if i in selected]
        if ambiguous:
            yield block if mode == "block_recheck" else ambiguous


class CompactReader(IsolatedReader):
    @torch.inference_mode()
    def evaluate(self, text, clauses, mode):
        start = perf_counter()
        front, suffixes = shared_prompt(self.tokenizer, text, clauses)
        schedule = list(blocks(front, suffixes))
        output = torch.empty((len(suffixes), len(ROLES)), dtype=torch.float32, device="cuda")
        slots, carried = 0, 0
        if mode != "independent":
            prefix = torch.tensor([front], device="cuda")
            cache = self.backbone(input_ids=prefix, use_cache=True).past_key_values
            if cache.get_seq_length() != len(front):
                raise RuntimeError("Incomplete context cache")
            slots += len(front)
        for block in schedule:
            positions, suffix_batch = zip(*block, strict=True)
            if mode == "independent":
                scores, cost = self.independent(front, suffix_batch)
            else:
                scores, cost = self.cached(front, suffix_batch, cache)
            output[list(positions)] = scores.float()
            slots += cost
        rechecked = recheck_calls = 0
        if mode in ("block_recheck", "row_recheck", "coalesced_recheck"):
            best = output.topk(2, dim=1).values
            selected = set(torch.nonzero(best[:, 0] - best[:, 1] <= CUTOFF).flatten().cpu().tolist())
            for batch in recheck_batches(front, suffixes, schedule, selected, mode):
                positions, suffix_batch = zip(*batch, strict=True)
                scores, cost = self.independent(front, suffix_batch)
                output[list(positions)] = scores.float()
                rechecked += len(batch)
                recheck_calls += 1
                slots += cost
            carried = len(selected)
        labels = output.argmax(1).cpu().numpy()
        return dict(labels=labels, ms=(perf_counter() - start) * 1000,
                    first_pass_blocks=len(schedule), flagged_rows=carried,
                    rechecked_rows=rechecked, recheck_calls=recheck_calls, forward_slots=slots)


def summarize(rows):
    result = {}
    for arm in ARMS:
        times = []
        outcomes = Counter()
        for row, mark, gold in rows:
            reference = row["independent"]["labels"]
            label = row[arm]["labels"]
            changed = label != reference
            outcomes["clauses"] += len(label)
            outcomes["changed_labels"] += int(changed.sum())
            outcomes["changed_requests"] += int(changed.any())
            outcomes["added_qualifications"] += int(((label == 2) & (reference != 2)).sum())
            outcomes["removed_qualifications"] += int(((label != 2) & (reference == 2)).sum())
            for field in ("first_pass_blocks", "flagged_rows", "rechecked_rows", "recheck_calls", "forward_slots"):
                outcomes[field] += row[arm][field]
            if mark is not None:
                outcomes["correct_targets"] += int(label[mark] == gold)
            times.append(row[arm]["ms"])
        result[arm] = dict(outcomes) | {"p50_p95_ms": np.quantile(times, [.5, .95]).tolist()}
    return dict(requests=len(rows), arms=result)


def run(development, controls, checkpoint, output, disagreements=None):
    if (output.exists() or (disagreements is not None and disagreements.exists())
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise RuntimeError("Require new outputs and Slurm CUDA")
    public = json.loads(controls.read_bytes())
    real = [json.loads(line) for line in development.read_bytes().splitlines()]
    if len(real) != 80 or any(row["split"] != "development" for row in real):
        raise ValueError("Use only the fixed development projection")
    reader = CompactReader(checkpoint)
    for arm in ARMS:
        reader.evaluate(public[0]["text"], spans(public[0]["text"]), arm)
    report, cases = {}, []
    for name, cohort in (("controls", public), ("development", real)):
        rows = []
        for i, item in enumerate(cohort):
            text = item["text"]
            clauses = spans(text)
            mark, gold = target(text, clauses, item)
            offset = i % len(ARMS)
            choices = {arm: reader.evaluate(text, clauses, arm) for arm in ARMS[offset:] + ARMS[:offset]}
            rows.append((choices, mark, gold))
            if name == "development" and disagreements is not None:
                for index, clause in enumerate(clauses):
                    labels = {arm: ROLES[choices[arm]["labels"][index]] for arm in ARMS}
                    if len(set(labels.values())) > 1:
                        cases.append(dict(query_id=item["id"], **clause, labels=labels))
            if (i + 1) % 20 == 0:
                print(name, i + 1, "of", len(cohort), flush=True)
        report[name] = summarize(rows)
    private_digest = None
    if disagreements is not None:
        with disagreements.open("x", encoding="utf-8") as handle:
            disagreements.chmod(0o600)
            json.dump(cases, handle, indent=2)
        private_digest = digest(disagreements)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=reader.provenance,
        script_sha256=digest(Path(__file__)), controls_sha256=digest(controls),
        development_sha256=digest(development), base_reader_sha256=digest(Path(__file__).with_name("enterprise_isolated_parsing.py")),
        cutoff=CUTOFF, report=report, private_disagreements_sha256=private_digest,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "controls", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--disagreements", type=Path)
    run(**vars(parser.parse_args()))
