"""Compare whole-block and row-selective qualification rechecks."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import linear

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval.enterprise_cross_precision import CONTROLS, NARROW, QUALIFICATION, evaluate
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import padded, spans
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write

ARMS = ("whole_block", "selected_rows", "selected_rows_fixed_width")
FIELDS = ("qualifications", "ranked", "initial", "decisions", "extra", "invoked", "clipped")


class Parser:
    def __init__(self, original):
        self.readers = original.readers

    def recheck(self, reader, front, suffixes, cached, alternate):
        self.counts.update(fallback_blocks=1, available_rows=len(suffixes))
        if self.arm == "whole_block":
            self.counts.update(executed_rows=len(suffixes))
            return reader.independent(front, suffixes)

        other = [index for index in range(cached.shape[1]) if index != QUALIFICATION]
        alternate_best = alternate[:, other].float().amax(dim=1)
        alt_gap = (alternate[:, QUALIFICATION].float() - alternate_best).abs()
        disagree = (cached.argmax(dim=1) == QUALIFICATION) != (alternate.argmax(dim=1) == QUALIFICATION)
        selected = torch.where(disagree | (alt_gap <= NARROW))[0]
        if not selected.numel():
            raise ValueError("The selected rows must explain the block trigger")
        self.counts.update(executed_rows=selected.numel())

        if self.arm == "selected_rows":
            scores, tokens = reader.independent(front, [suffixes[i] for i in selected.tolist()])
        else:
            ids, mask, positions = padded([front + item for item in suffixes], reader.tokenizer.pad_token_id)
            ids, mask, positions = (tensor.index_select(0, selected) for tensor in (ids, mask, positions))
            hidden = reader.backbone(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
            scores = linear(hidden.last_hidden_state[:, -1].contiguous(), reader.head)
            tokens = ids.numel()

        result = cached.clone()
        result.index_copy_(0, selected, scores)
        return result, tokens

    @torch.inference_mode()
    def evaluate(self, text, clauses, method, cutoff):
        if method != "broadcast_prefix" or cutoff != 2.:
            raise ValueError("Unexpected qualification parser contract")
        self.counts = Counter()
        for reader in self.readers.values():
            reader.broadcast = True
            reader.forks = reader.prefix_bytes = 0
        started = perf_counter()
        result = evaluate(text, clauses, self.readers, "consensus", recheck=self.recheck)
        self.ms = (perf_counter()-started) * 1000
        self.counts.update(fallback_slots=result.get("fallback_slots", 0),
                           fp16_checks=result.get("fp16_block_checks", 0))
        self.labels = result["labels"]
        return dict(labels=self.labels, rechecked=result.get("rechecked", 0))


def order(index, reverse):
    arms = ARMS[::-1] if reverse else ARMS
    shift = index % len(arms)
    return arms[shift:] + arms[:shift]


def summarize(observations):
    baseline = [row["whole_block"] for row in observations]
    sample = np.random.default_rng(20261121).integers(len(baseline), size=(6000, len(baseline)))
    baseline_timing = np.asarray([row["elapsed_ms"] for row in baseline])
    report = {}
    for arm in ARMS:
        current = [row[arm] for row in observations]
        ms = np.asarray([row["elapsed_ms"] for row in current])
        paired = np.quantile(ms[sample], .95, axis=1) - np.quantile(baseline_timing[sample], .95, axis=1)
        report[arm] = dict(
            parser_work=dict(sum((Counter(row["work"]) for row in current), Counter())),
            parser_p50_p95_ms=np.quantile([row["parser_ms"] for row in current], [.50,.95]).tolist(),
            live_p50_p95_ms=np.quantile(ms, [.50,.95]).tolist(),
            paired_live_p95_delta_ci95_ms=np.quantile(paired, [.025,.975]).tolist(),
            issued=sum(bool(row["initial"] or row["extra"]) for row in current),
            changed={key: sum(row[key] != ref[key] for row, ref in zip(current, baseline, strict=True))
                     for key in (*FIELDS, "labels")},
        )
    return report


def run(queries, corpus, roles, vectors, meta, models, serial, controls, output, reverse):
    require_cuda(output)
    if digest(controls) != CONTROLS:
        raise ValueError("Use the pinned bilingual public controls")
    texts, fixture, fingerprint = init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    parser = Parser(fixture[0])
    staged = (parser, *fixture[1:])
    observations = []
    for index, text in enumerate(texts):
        reference = packed.one(text, "guarded_rescue", *fixture)
        variants = {}
        for arm in order(index, reverse):
            parser.arm = arm
            row = packed.one(text, "guarded_rescue", *staged)
            variants[arm] = dict(row, labels=tuple(map(int, parser.labels)),
                                 parser_ms=parser.ms, work=dict(parser.counts))
        if any(variants["whole_block"][key] != reference[key] for key in FIELDS):
            raise ValueError("Instrumented whole-block path changed the pipeline reference")
        observations.append(variants)
        if (index + 1) % 20 == 0:
            print(index + 1, "development requests measured", flush=True)

    results = {arm: Counter(targets=0) for arm in ARMS}
    public = json.loads(controls.read_bytes())
    for index, row in enumerate(public):
        text = row["text"]
        clauses = spans(text)
        position, gold = target(text, clauses, row)
        if position is None:
            raise ValueError("Missing control target")
        for arm in order(index, reverse):
            parser.arm = arm
            predicted = int(parser.evaluate(text, clauses, "broadcast_prefix", 2.)["labels"][position])
            results[arm].update(targets=1, correct=predicted == gold,
                                false_qualification=predicted == QUALIFICATION and gold != QUALIFICATION,
                                missed_qualification=predicted != QUALIFICATION and gold == QUALIFICATION)
    write(output, dict(
        slurm_job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)), reverse=reverse,
        controls_sha256=CONTROLS, controls=results, development=summarize(observations), **fingerprint,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "controls", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
