"""Limit independent enterprise parser rechecks to contested clause rows."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import enterprise_shared_qualification as search
from experiments.retrieval.enterprise_cross_precision import CONTROLS, NARROW, QUALIFICATION, WIDE, evaluate
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_guarded_parsing import target
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("whole_block", "clause_narrow", "clause_wide")


def qualification_lead(values):
    values = values.float()
    other = [idx for idx in range(values.shape[1]) if idx != QUALIFICATION]
    return values[:, QUALIFICATION] - values[:, other].amax(dim=1)


class ClauseParser(ComparisonParser):
    mode = "whole_block"

    def recheck(self, reader, front, suffixes, first, second):
        if second is None:
            raise ValueError("Expected the original two-precision consensus")
        if self.mode == "whole_block":
            picked = list(range(len(suffixes)))
        else:
            a, b = qualification_lead(first), qualification_lead(second)
            disagrees = (first.argmax(dim=1) == QUALIFICATION) != (second.argmax(dim=1) == QUALIFICATION)
            risk = (b.abs() <= NARROW) | disagrees
            if self.mode == "clause_wide":
                risk |= a.abs() <= WIDE
            picked = risk.nonzero().flatten().tolist()
        if not picked:
            raise ValueError("Original fallback trigger has no contested row")
        confirmed, slots = reader.independent(front, [suffixes[i] for i in picked])
        out = first.clone()
        out[picked] = confirmed
        self.run_counts.update(fallback_rows=len(picked), original_block_rows=len(suffixes), fallback_slots=slots)
        return out, slots

    @torch.inference_mode()
    def evaluate(self, text, clauses, method, cutoff):
        if method != "broadcast_prefix" or cutoff != 2.:
            raise ValueError("Unexpected parser contract")
        began = perf_counter()
        for reader in self.readers.values():
            reader.broadcast = True
            reader.forks = reader.prefix_bytes = 0
        self.run_counts = Counter()
        scores = evaluate(text, clauses, self.readers, "consensus", recheck=self.recheck)
        self.labels = tuple(map(int, scores["labels"]))
        self.run_counts.update(rechecked_blocks=scores.get("rechecked", 0))
        self.ms = (perf_counter() - began) * 1000
        return dict(labels=scores["labels"], rechecked=scores.get("rechecked", 0))


def request(text, arm, fixture):
    parser = fixture[0]
    parser.mode = arm
    result = search.one(text, "automatic", *fixture)
    return dict(result, labels=parser.labels, parser_ms=parser.ms, parser_counts=dict(parser.run_counts))


def control(row, parser, arm):
    parser.mode = arm
    clauses = spans(row["text"])
    index, gold = target(row["text"], clauses, row)
    parser.evaluate(row["text"], clauses, "broadcast_prefix", 2.)
    predicted = parser.labels[index]
    return dict(labels=parser.labels, target_correct=predicted == gold,
                false_qualification=predicted == QUALIFICATION and gold != QUALIFICATION,
                missed_qualification=predicted != QUALIFICATION and gold == QUALIFICATION)


def compare(observations, controls=False):
    earlier = [r["whole_block"] for r in observations]
    draws = np.random.default_rng(20261124).integers(len(earlier), size=(6000, len(earlier)))
    result = {}
    for arm in ARMS:
        current = [r[arm] for r in observations]
        matched = list(zip(earlier, current, strict=True))
        report = result[arm] = dict(
            queries=len(current),
            requests_with_role_changes=sum(a["labels"] != b["labels"] for a, b in matched),
            changed_qualification_membership=sum(
                tuple(v == QUALIFICATION for v in a["labels"]) != tuple(v == QUALIFICATION for v in b["labels"])
                for a, b in matched))
        if controls:
            report.update({key: sum(row[key] for row in current)
                           for key in ("target_correct", "false_qualification", "missed_qualification")})
            continue
        times = np.asarray([r["elapsed_ms"] for r in current])
        previous = np.asarray([r["elapsed_ms"] for r in earlier])
        difference = np.quantile(times[draws], .95, axis=1) - np.quantile(previous[draws], .95, axis=1)
        report.update(
            issued=sum(bool(r["initial"] or r["extra"]) for r in current),
            parser_counts=dict(sum((Counter(row["parser_counts"]) for row in current), Counter())),
            parser_p50_p95_ms=np.quantile([r["parser_ms"] for r in current], [.5, .95]).tolist(),
            live_p50_p95_ms=np.quantile(times, [.5, .95]).tolist(),
            paired_live_p95_delta_ci95=np.quantile(difference, [.025, .975]).tolist(),
            changes={key: sum(a[key] != b[key] for a, b in matched)
                     for key in ("qualifications", "retrieved", "ranked", "decisions", "initial", "extra", "invoked")})
    return result


def run(queries, corpus, roles, vectors, metadata, models, control_file, output):
    search.role.require_cuda(output)
    if (digest(queries), digest(corpus), digest(control_file)) != (DEVELOPMENT, SOURCE_INVENTORY, CONTROLS):
        raise ValueError("Unexpected private or public inputs")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {row["id"]: row for row in rows(corpus)}
    meta = json.loads(metadata.read_bytes())
    enc = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(enc)
    if (SOURCE_INVENTORY, provenance["revision"], digest(vectors), len(sources)) != tuple(
            meta[k] for k in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Vector index provenance differs")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    parser = ClauseParser(models / "qwen3-4b-instruct-2507")
    verifier = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    fixture = (parser, QwenTorchEmbedding(enc), index, sources,
               search.role.load_role_index(roles, SOURCE_INVENTORY, sources), verifier)
    for arm in ARMS:
        request("Find an expert in sensor calibration. Prior testing on optical components is required.", arm, fixture)
    observed = []
    for i, row in enumerate(rows(queries)):
        offset = i % len(ARMS)
        observed.append({arm: request(row["text"], arm, fixture) for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1} private development requests", flush=True)
    public = [{arm: control(row, parser, arm) for arm in ARMS} for row in json.loads(control_file.read_bytes())]
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       code_sha256=digest(Path(__file__)), controls=compare(public, True),
                       live=compare(observed)))


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "control_file", "output"):
        arg.add_argument(name, type=Path)
    run(**vars(arg.parse_args()))
