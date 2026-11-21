"""Compare independent full-context layouts for triggered parser rechecks."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import linear
from transformers import AttentionInterface

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention
from experiments.retrieval.cof_ragged_shared_weights import packed_view
from experiments.retrieval.enterprise_clause_recheck import ClauseParser, control, qualification_lead
from experiments.retrieval.enterprise_cross_precision import CONTROLS, NARROW
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write

ARMS = ("whole_block", "varlen", "four_rows", "agreement")
FIELDS = ("initial", "extra", "ranked", "qualifications", "decisions", "clipped", "invoked")


class LayoutParser(ClauseParser):
    def __init__(self, reference):
        self.readers = reference.readers
        reader = self.readers["bf16"]
        AttentionInterface.register(PACKED_ATTENTION, packed_attention)
        self.ragged = packed_view(reader.backbone, share_weights=True)
        if any(a.data_ptr() != b.data_ptr() for a, b in zip(
                self.ragged.parameters(), reader.backbone.parameters(), strict=True)):
            raise ValueError("The counterfactual layout must share the reference weights")

    def variable_length(self, reader, front, suffixes):
        sequences = [front + suffix for suffix in suffixes]
        lengths = [len(sequence) for sequence in sequences]
        limits = [0, *np.cumsum(lengths).tolist()]
        tokens = torch.tensor([[token for seq in sequences for token in seq]], device="cuda")
        positions = torch.tensor([[i for length in lengths for i in range(length)]], device="cuda")
        ends = torch.tensor([i-1 for i in limits[1:]], device="cuda")
        offsets = torch.tensor(limits, device="cuda", dtype=torch.int32)
        hidden = self.ragged(input_ids=tokens, position_ids=positions,
                             attention_mask={"full_attention": None}, offsets=offsets,
                             max_length=max(lengths), use_cache=False).last_hidden_state[0]
        return linear(hidden.index_select(0, ends).contiguous(), reader.head), limits[-1]

    def four_rows(self, reader, front, suffixes):
        scores, counts = zip(*(reader.independent(front, suffixes[i:i+4])
                               for i in range(0, len(suffixes), 4)), strict=True)
        return torch.cat(scores), sum(counts)

    def recheck(self, reader, front, suffixes, first, second):
        started = perf_counter()
        if self.mode == "whole_block":
            result = super().recheck(reader, front, suffixes, first, second)
        else:
            widths = len(front) + max(map(len, suffixes))
            self.run_counts.update(original_block_rows=len(suffixes), canonical_token_slots=len(suffixes)*widths)
            if self.mode == "four_rows":
                result = self.four_rows(reader, front, suffixes)
            else:
                result = self.variable_length(reader, front, suffixes)
            if self.mode == "agreement":
                alternate, more = self.four_rows(reader, front, suffixes)
                margins = qualification_lead(result[0])
                alternate_margins = qualification_lead(alternate)
                accepted = ((margins > 0) == (alternate_margins > 0)).all() & (
                    torch.minimum(margins.abs(), alternate_margins.abs()) > NARROW).all()
                self.run_counts.update(fallback_slots=more, agreement_attempts=1)
                if not bool(accepted.item()):
                    self.run_counts.update(fallback_slots=result[1], agreement_rejects=1)
                    result = reader.independent(front, suffixes)
            self.run_counts.update(fallback_rows=len(suffixes), fallback_slots=result[1])
        self.run_counts["fallback_ms"] += (perf_counter()-started) * 1000
        return result


def search(text, mode, fixture):
    parser = fixture[0]
    parser.mode = mode
    result = packed.one(text, "guarded_rescue", *fixture)
    return result | dict(labels=parser.labels, parser_ms=parser.ms, parser_work=dict(parser.run_counts))


def report(observations, public):
    reference = np.asarray([row[ARMS[0]]["elapsed_ms"] for row in observations])
    sample = np.random.default_rng(20261121).integers(len(reference), size=(6000, len(reference)))
    output = {}
    for arm in ARMS:
        records = [row[arm] for row in observations]
        times = np.asarray([x["elapsed_ms"] for x in records])
        issued = sum(bool(x["initial"] or x["extra"]) for x in records)
        diffs = {field: sum(x[arm][field] != x[ARMS[0]][field] for x in observations)
                 for field in (*FIELDS, "labels")}
        control_changes = sum(x[arm]["labels"] != x[ARMS[0]]["labels"] for x in public)
        paired = np.quantile(times[sample], .95, axis=1) - np.quantile(reference[sample], .95, axis=1)
        stages = [r for r in records if r["parser_work"].get("rechecked_blocks", 0)]
        counts = sum((Counter(r["parser_work"]) for r in records), Counter())
        output[arm] = dict(issued=issued, changes=diffs, parser_work=dict(counts),
            parse_p50_p95_ms=np.quantile([x["parser_ms"] for x in records], [.5, .95]).tolist(),
            search_p50_p95_ms=np.quantile(times, [.5, .95]).tolist(),
            recheck_requests=len(stages),
            recheck_request_search_p50_ms=float(np.median([x["elapsed_ms"] for x in stages])) if stages else None,
            paired_search_p95_delta_ci95=np.quantile(paired, [.025, .975]).tolist(),
            synthetic_controls=dict(any_label_changes=control_changes, **{key: sum(x[arm][key] for x in public)
                for key in ("target_correct", "false_qualification", "missed_qualification")}))
    return output


def run(queries, corpus, roles, vectors, meta, models, serial, controls, output, reverse):
    require_cuda(output)
    if digest(controls) != CONTROLS:
        raise ValueError("Expected the pinned public controls")
    texts, fixture, fingerprint = init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    parser = LayoutParser(fixture[0])
    alternate_fixture = (parser, *fixture[1:])
    reader = parser.readers["bf16"]
    first = reader.tokenizer.encode("Optical experience", add_special_tokens=False)
    second = reader.tokenizer.encode("Prior experience is mandatory", add_special_tokens=False)
    front = reader.tokenizer.encode("Classify clause", add_special_tokens=False)
    with torch.inference_mode():
        original = parser.variable_length(reader, front, [first, second])[0]
        mutated = parser.variable_length(reader, front, [first[::-1], second])[0]
        if not torch.equal(original[1], mutated[1]):
            raise ValueError("Changing one suffix influenced a neighbor")
    warmup = "Find an expert in optical inspection. Prior testing with thin films is required."
    for arm in ARMS:
        search(warmup, arm, alternate_fixture)
    observations = []
    ordered = ARMS[::-1] if reverse else ARMS
    for i, text in enumerate(texts):
        reference = packed.one(text, "guarded_rescue", *fixture)
        sequence = ordered[i%len(ordered):] + ordered[:i%len(ordered)]
        rows = {arm: search(text, arm, alternate_fixture) for arm in sequence}
        if any(reference[key] != rows[ARMS[0]][key] for key in FIELDS):
            raise ValueError("The timed reference diverged from the existing stack")
        observations.append(rows)
        if (i+1)%20 == 0:
            print(i+1, "development requests", flush=True)
    public = [{arm: control(row, parser, arm) for arm in ordered}
              for row in json.loads(controls.read_bytes())]
    signature = sha256(json.dumps([{arm: {
        **{field: row[arm][field] for field in (*FIELDS, "labels")},
        "recheck_counts": {key: value for key, value in row[arm]["parser_work"].items() if key != "fallback_ms"},
    } for arm in ARMS} for row in observations], sort_keys=True).encode()).hexdigest()
    write(output, dict(arms=report(observations, public), slurm_job=os.environ["SLURM_JOB_ID"],
        reverse=reverse, trials=len(public), output_signature=signature, controls_sha256=digest(controls),
        runner_sha256=digest(Path(__file__)), **fingerprint))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "controls", "output"):
        cli.add_argument(key, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
