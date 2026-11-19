"""Test whether author-scoped role support helps select an intended source pair."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding import TASK
from experiments.credit import shared_embedding_witness_gate as gate
from experiments.credit.batched_source_role_search import SourceIndex, declared_roles
from experiments.credit.scoped_credit_search import DENSE_TASK, EvidenceIndex, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


FROZEN = {
    "projection": "162579864f25bea0f2ad422737e16395272d0c2b58b85f53ada23cbb745f49a1",
    "panel": "86f35bd6d07fae2985b5cb68a54967293821cb7169bd15980276eec70a55a37f",
    "boundary_words": "34f9163daaf4c321c8fe398c493fed5f982adc3e2200cb61ee1881bb2fa9851c",
    "prior": "d0231fc2fe81c77761be4ab1fa6c6c656e6fd8770f3d6b2eb09ff1febad160ee",
    "glossary": gate.replay.EXAMPLES,
}
ARMS = ("source_top4", "source_first", "source_backfill", "pooled_priority", "owner_priority")
CONDITIONS = ("both_present", "missing_left", "missing_right")


def pairs(index, ranks):
    right = {source: rank for rank, source in enumerate(ranks[1][:12], 1)}
    options = []
    for i, x in enumerate(ranks[0][:12], 1):
        for y in index.links[x] & right.keys():
            j = right[y]
            if min(i, j) <= 4:
                options.append((i + j, max(i, j), x, y))
    return [(maximum, index.dois[x], index.dois[y]) for _, maximum, x, y in sorted(options)]


def fitting_owners(evidence, pair, tasks, role_scores, cutoff):
    left, right = [evidence.roles[doi] for doi in pair]
    common = left.keys() & right.keys()
    if not common:
        return set(), False
    sides = (left, right)
    # Open each pair's indexed declarations once. The pooled ablation can
    # combine different people at proposal time, but still verifies ownership.
    accepted = [
        {owner for owner in common if gate.membership_margin(
            scores, declared_roles(sides[side][owner])) > cutoff}
        for (side, _), scores in zip(tasks, role_scores, strict=True)
    ]
    return set.intersection(*accepted), all(accepted)


def select(options, evidence, tasks, scores, cutoff, arm):
    if arm == "source_top4":
        options = [row for row in options if row[0] <= 4]
    if arm in {"source_top4", "source_first"}:
        options = options[:1]
    if arm == "source_backfill":
        options = options[:2]

    reads = 0
    for _, *sources in options:
        pair = tuple(sources)
        owners, pooled = fitting_owners(evidence, pair, tasks, scores, cutoff)
        reads += 1
        if owners:
            return pair, owners, reads, False
        if arm == "pooled_priority" and pooled:
            return None, set(), reads, True
    return None, set(), reads, False


def run(projection, panel, glossary, boundary_words, training, prior, embedding, output):
    files = dict(projection=projection, panel=panel, glossary=glossary,
                 boundary_words=boundary_words, prior=prior)
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm CUDA node")
    for name, path in files.items():
        if gate.replay.digest(path) != FROZEN[name]:
            raise ValueError(f"Changed {name} input")
    corpus, clues, inventory, boundary, lock = [json.loads(path.read_text()) for path in files.values()]
    provenance = verify_checkpoint(embedding)
    if (clues["frozen_sha256"] != FROZEN["projection"]
            or lock["embedding_revision"] != provenance["revision"]
            or lock["training_sha256"] != gate.replay.digest(training)):
        raise ValueError("Changed corpus, embedding or training data")

    documents = corpus["documents"]
    cases, links = gate.replay.panel(documents)
    components = {(left, right): component for left, right, component in clues["pairs"]}
    if not all(pair in components for pair, *_ in cases) or links != len(components):
        raise ValueError("Changed publisher link graph")
    niso_phrases = defaultdict(set)
    for row in inventory["examples"]:
        niso_phrases[row["role"]].add(row["text"])
    wording = {"niso_examples": niso_phrases, "external_wording": boundary}
    encoder = gate.replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    bank = gate.build_banks(encoder, inventory, training)[0]["ridge"]
    cutoff = lock["thresholds"]["ridge"]["caps"]["10"]["threshold"]
    evidence = EvidenceIndex(documents)
    counts, times = defaultdict(Counter), defaultdict(list)
    quality_by_case = defaultdict(dict)
    source_pool, family_outcomes = defaultdict(Counter), defaultdict(dict)
    supported_components = defaultdict(set)

    with TemporaryDirectory() as directory:
        index = SourceIndex(corpus, encoder, Path(directory))
        for i, (gold_pair, family, variant, tasks, gold_owners) in enumerate([cases[0], *cases]):
            for word_kind, phrases in wording.items():
                words = [
                    min(phrases[role], key=lambda phrase: gate.replay.stable(family, role, phrase))
                    for _, role in tasks
                ]
                for view in ("result", "keywords"):
                    source_texts = [clues["clues"][doi][view] for doi in gold_pair]
                    prompts = [f"Instruct: {DENSE_TASK}\nQuery: {text}" for text in source_texts]
                    prompts.extend(f"Instruct: {TASK}\nQuery:{word}" for word in words)
                    started = perf_counter()
                    vectors = encoder.encode(prompts)
                    scores = vectors[2:] @ bank.T
                    encoding_ms = (perf_counter() - started) * 1000
                    for condition in CONDITIONS:
                        removed = dict(missing_left=gold_pair[:1], missing_right=gold_pair[1:]).get(condition, ())
                        started = perf_counter()
                        _, fused = index.rank_sources(source_texts, vectors[:2], removed)
                        options = pairs(index, fused)
                        prefix_ms = encoding_ms + (perf_counter() - started) * 1000
                        arm_order = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]
                        key = word_kind, view, condition
                        if i:
                            listed = any((a, b) == gold_pair for _, a, b in options)
                            eligible = bool(gold_owners and not removed)
                            oracle_owners, _ = fitting_owners(evidence, gold_pair, tasks, scores, cutoff) if eligible else (set(), False)
                            source_pool[key].update(queries=1, pairs=len(options),
                                                    empty=not options, intended_pair_listed=listed,
                                                    positive_queries=eligible,
                                                    positive_listed=eligible and listed,
                                                    gate_recovers_given_intended_source=bool(oracle_owners & gold_owners),
                                                    recoverable_in_pool=listed and bool(oracle_owners & gold_owners))
                        for arm in arm_order:
                            started = perf_counter()
                            pair, owners, reads, pooled_error = select(options, evidence, tasks, scores, cutoff, arm)
                            elapsed = prefix_ms + (perf_counter() - started) * 1000
                            if not i:
                                continue
                            expected = gold_owners if not removed else set()
                            supported = owners & expected if pair == gold_pair else set()
                            wrong = owners - supported
                            exact = supported == expected and not wrong
                            counts[key, arm].update(queries=1, supported_queries=bool(supported),
                                                   unsupported_queries=bool(wrong),
                                                   wrong_source_queries=pair is not None and pair != gold_pair,
                                                   wrong_author_queries=pair == gold_pair and bool(wrong),
                                                   exact=exact, inspected_pairs=reads,
                                                   stopped_at_pooled_only=pooled_error)
                            if supported:
                                supported_components[key, arm].add(components[gold_pair])
                            times[key, arm].append(elapsed)
                            quality_by_case[key, arm][family, variant] = (bool(supported), bool(wrong))
                            family_outcomes[key, arm].setdefault(family, []).append(exact)

    report = {}
    for key, pool in source_pool.items():
        arms = {}
        baseline = quality_by_case[key, "source_first"]
        for arm in ARMS:
            current = quality_by_case[key, arm]
            arms[arm] = dict(
                counts[key, arm],
                components_with_supported_query=len(supported_components[key, arm]),
                both_edits_exact=sum(len(group) == 2 and all(group) for group in family_outcomes[key, arm].values()),
                supported_gains=sum(current[item][0] and not value[0] for item, value in baseline.items()),
                supported_losses=sum(value[0] and not current[item][0] for item, value in baseline.items()),
                unsupported_gains=sum(current[item][1] and not value[1] for item, value in baseline.items()),
                unsupported_losses=sum(value[1] and not current[item][1] for item, value in baseline.items()),
                p95_ms=float(np.quantile(times[key, arm], .95)),
            )
        report["/".join(key)] = dict(pool=pool, arms=arms)

    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            inputs=FROZEN, training_sha256=gate.replay.digest(training),
                            script_sha256=gate.replay.digest(Path(__file__)), cutoff=cutoff,
                            embedding_revision=provenance["revision"], documents=len(documents), linked_pairs=links,
                            tested_pairs=len({pair for pair, *_ in cases}), cases=len(cases), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "panel", "glossary", "boundary_words", "training", "prior", "embedding", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
