"""Stress-test a frozen lexical expansion gate on natural literature queries."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_clue_agreement import lexical_coverage
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.dense_retrieval import FUSION_DEPTH, Search, target_groups
from experiments.litsearch.source_withholding import QUERY_SHA256, REVISION, split
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


NARROW = 4
FROZEN_HASH = "8c0df56892eed0879d5b46c29dceed249beade0f818a33569abd6ad188ebc45b"
MODES = ("top4_agreement", "expanded_agreement", "plos_cutoff", "local_strict")


def agree(sparse, dense, depth):
    positions = {doc: i for i, doc in enumerate(dense[:depth], 1)}
    possible = [(1 / (60 + i) + 1 / (60 + positions[doc]), -doc, doc)
                for i, doc in enumerate(sparse[:depth], 1) if doc in positions and min(i, positions[doc]) <= NARROW]
    return max(possible)[2] if possible else None


def observe(endpoint, question, hidden, df, terms):
    text = question["query"]
    start = perf_counter()
    sparse = endpoint.lexical(text, hidden, FUSION_DEPTH)
    dense = endpoint.dense(text, hidden, FUSION_DEPTH)
    narrow, broad = (agree(sparse, dense, depth) for depth in (NARROW, FUSION_DEPTH))
    base = (perf_counter() - start) * 1000
    score, extra = None, 0.0
    if narrow is None and broad is not None:
        start = perf_counter()
        weights = {token: float(np.log((1 + len(terms)) / (1 + df[token]))) for token in set(tokens(text))}
        score = lexical_coverage(text, terms[endpoint.positions[broad]], weights)
        extra = (perf_counter() - start) * 1000
    if (narrow in hidden) or (broad in hidden):
        raise ValueError("Reference target leaked into a withheld ranking")
    return dict(narrow=narrow, broad=broad, score=score, base_ms=base, gate_ms=extra)


def replay(endpoint, questions, df, doc_terms):
    output = []
    for i, q in enumerate(questions):
        targets = set(q["corpusids"])
        if not targets <= endpoint.positions.keys():
            raise ValueError("A reference paper is missing")
        views = ("present", "withheld")
        output.append(dict(targets=q["corpusids"], category=q["query_set"], views={
            view: observe(endpoint, q, targets if view == "withheld" else set(), df, doc_terms)
            for view in views[i % 2:] + views[:i % 2]
        }))
    return output


def choose(obs, policy, cutoffs):
    if policy == "expanded_agreement":
        return obs["broad"], obs["base_ms"]
    if policy == "top4_agreement" or obs["score"] is None:
        return obs["narrow"], obs["base_ms"]
    return (obs["broad"] if obs["score"] > cutoffs[policy] else None), obs["base_ms"] + obs["gate_ms"]


def summarize(rows, cutoffs):
    result = {}
    groups = target_groups(rows)
    for policy in MODES:
        result[policy] = {}
        for view in ("present", "withheld"):
            selected, counts, timings = [], Counter(), []
            for row in rows:
                obs = row["views"][view]
                selected_id, elapsed = choose(obs, policy, cutoffs)
                matched = selected_id in row["targets"]
                counts.update(dict(returns=selected_id is not None, gold_returned=matched,
                                   nonreference_returns=selected_id is not None and not matched,
                                   expansion_candidates=obs["score"] is not None,
                                   expansion_reference_candidates=obs["score"] is not None and obs["broad"] in row["targets"],
                                   expansion_reference_returns=obs["score"] is not None and matched,
                                   expansion_nonreference_returns=obs["score"] is not None and selected_id is not None and not matched))
                selected.append(int(matched if view == "present" else selected_id is not None))
                timings.append(elapsed)
            report = dict(counts, n=len(rows), p95_ms=float(np.quantile(timings, .95)))
            if policy != "top4_agreement":
                base = [int(row["views"][view]["narrow"] in row["targets"] if view == "present" else
                            row["views"][view]["narrow"] is not None) for row in rows]
                report["paired_delta_vs_top4"] = clustered_paired_interval(
                    [value - before for value, before in zip(selected, base, strict=True)], groups
                )
            result[policy][view] = report
    return result


def run(projection, cache_file, embedding, freeze_file, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if digest(freeze_file) != FROZEN_HASH:
        raise ValueError("Expected the prior PLOS threshold")
    freeze = json.loads(freeze_file.read_text())
    projection_hash = digest(projection)
    with gzip.open(projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch revision")
    papers = data["papers"]
    provenance = verify_checkpoint(embedding)
    if provenance["revision"] != freeze["embedding_revision"]:
        raise ValueError("Changed encoder revision")
    with np.load(cache_file, allow_pickle=False) as cached:
        expected = [r["corpusid"] for r in papers]
        if (cached["corpus_ids"].tolist() != expected or str(cached["projection_sha256"]) != projection_hash
                or json.loads(str(cached["provenance"])) != provenance):
            raise ValueError("Frozen vectors do not match this corpus")
        vectors = torch.from_numpy(cached["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    mapping, terms, df = {}, [], Counter()
    for i, paper in enumerate(papers):
        value = paper_text(paper)
        words = set(tokens(value))
        terms.append(words)
        df.update(words)
        while value in mapping:
            value += "\n"
        mapping[value] = i
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    validation, test, eligible = split(data["questions"])
    with TemporaryDirectory() as tmp:
        index = build_lexical(mapping, Path(tmp))
        endpoint = Search(papers, index, vectors, encoder, None)
        observe(endpoint, validation[0], set(), df, terms)
        development = replay(endpoint, validation, df, terms)
        false_candidates = [obs["score"] for row in development for obs in row["views"].values()
                            if obs["score"] is not None and obs["broad"] not in row["targets"]]
        cuts = dict(plos_cutoff=freeze["threshold"], local_strict=max(false_candidates))
        print("Calibrated with", len(false_candidates), "nonreference expansion candidates", flush=True)
        tested = replay(endpoint, test, df, terms)
    report = {"all": summarize(tested, cuts)}
    for category in ("manual_acl", "manual_iclr"):
        report[category] = summarize([row for row in tested if row["category"] == category], cuts)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), corpus=len(papers),
                            script_sha256=digest(Path(__file__)), projection_sha256=projection_hash,
                            cache_sha256=digest(cache_file), frozen_sha256=digest(freeze_file),
                            embedding_revision=provenance["revision"], eligible_calibration=eligible,
                            calibration_nonreference_expansions=len(false_candidates), thresholds=cuts,
                            calibration=summarize(development, cuts), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache_file", "embedding", "freeze_file", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
