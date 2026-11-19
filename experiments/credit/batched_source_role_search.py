"""Measure shared encoding and staged verification in scoped expert search."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding import TASK as EXPERT_TASK
from experiments.credit import scoped_credit_search as discovery
from experiments.credit import shared_embedding_witness_gate as gate
from experiments.credit.scoped_credit_language import scope_atoms
from experiments.credit.source_scoped_credit import ROLE_BITS


ARMS = ("separate", "joint_batch", "source_first")
SOURCE_POLICIES = ("linked_top4", "linked_expand")


class SourceIndex(discovery.Search):
    def rank_sources(self, clues, embeddings, removed):
        banned = {self.ids[doi] for doi in removed}
        fused = []
        dense = []
        for query, vector in zip(clues, embeddings, strict=True):
            order = [int(i) for i in np.argsort(-(self.vectors @ vector), kind="stable") if i not in banned]
            lexical = {int(i): value for i, value in self.lexical.document_scores(query).items() if int(i) not in banned}
            lex_order = sorted(lexical, key=lambda i: (-lexical[i], i))
            scores = {i: 1 / (60 + rank) for rank, i in enumerate(order, 1)}
            for rank, i in enumerate(lex_order, 1):
                scores[i] += 1 / (60 + rank)
            dense.append(order)
            fused.append(sorted(scores, key=lambda i: (-scores[i], i)))
        return dense, fused


@lru_cache(maxsize=1024)
def declared_roles(mask):
    return frozenset(role for role, bit in ROLE_BITS.items() if mask & bit)


def witness(index, sources, scopes, scores, cutoff):
    if sources is None:
        return []
    left, right = sources
    common = index.roles[left].keys() & index.roles[right].keys()
    return sorted(owner for owner in common if all(
        gate.membership_margin(row, declared_roles(index.roles[sources[slot]][owner])) > cutoff
        for slot, row in zip(scopes, scores, strict=True)
    ))


def forward(index, evidence, prompts, clues, scopes, removed, bank, cutoff, policy, arm):
    start = perf_counter()
    if arm == "joint_batch":
        encoded = index.encoder.encode(prompts)
        source, task = encoded[:2], encoded[2:]
        calls = 1
    else:
        source = index.encoder.encode(prompts[:2])
        task = index.encoder.encode(prompts[2:]) if arm == "separate" else None
        calls = 2 if task is not None else 1
    candidate = index.rank(*index.rank_sources(clues, source, removed), policy)
    if arm == "source_first" and candidate is not None:
        task = index.encoder.encode(prompts[2:])
        calls += 1
    found = witness(evidence, candidate, scopes, task @ bank.T if task is not None else None, cutoff)
    payload = json.dumps(dict(sources=candidate, owners=found), separators=(",", ":"))
    return candidate, set(found), calls, (perf_counter() - start) * 1000, payload


def count(outcomes, item, removed, evidence):
    selected, owners, calls, _, _ = outcomes
    expected_sources = tuple(item["dois"])
    expected = set() if removed else set(item["owners"])
    usable = owners if selected == expected_sources and not removed else set()
    supported = usable & expected
    wrong = owners - supported
    known = set() if selected is None else set(discovery.attributed(evidence, selected, item["atoms"]))
    return dict(requests=1, encoder_calls=calls, exact_sources=selected == expected_sources,
                wrong_source_pair=selected is not None and selected != expected_sources,
                unsupported_return=bool(wrong), supported_return=bool(supported),
                unsupported_owners=len(wrong), all_owners_and_sources=selected == expected_sources and owners == expected,
                known_roles_exact=selected == expected_sources and known == expected,
                available_positive=bool(expected),
                correct_no_support_with_exact_pair=selected == expected_sources and not owners and not expected)


def run(fixture_path, glossary_path, projection_path, training, embedding, prior, output):
    import torch

    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or discovery.digest(fixture_path) != discovery.FIXTURE
            or discovery.digest(glossary_path) != discovery.NISO
            or discovery.digest(projection_path) != discovery.PROJECTION):
        raise ValueError("Require the pinned fixture and fresh Slurm CUDA")
    reference = json.loads(prior.read_text())
    if (reference["training_sha256"] != discovery.digest(training)
            or reference["script_sha256"] != discovery.digest(Path(gate.__file__))
            or reference["projection_sha256"] != discovery.PROJECTION
            or reference["glossary_sha256"] != discovery.NISO):
        raise ValueError("Expected the published shared-embedding gate")
    provenance = gate.replay.verify_checkpoint(embedding)
    if provenance["revision"] != reference["embedding_revision"]:
        raise ValueError("Changed embedding revision")
    fixture, glossary, payload = (json.loads(p.read_text()) for p in (fixture_path, glossary_path, projection_path))
    items = [item for item in discovery.examples(fixture, glossary) if item["form"] == "explicit"]
    encoder = gate.replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    banks, offline_ms = gate.build_banks(encoder, glossary, training)
    cutoff = reference["thresholds"]["ridge"]["caps"]["10"]["threshold"]
    timings, counts = defaultdict(list), defaultdict(Counter)
    unique_sources = defaultdict(dict)
    pair_exact = defaultdict(lambda: defaultdict(list))
    evidence = discovery.EvidenceIndex(payload["documents"])
    with TemporaryDirectory() as directory:
        index = SourceIndex(payload, encoder, Path(directory))
        for i, item in enumerate([items[0], *items]):
            anchors = scope_atoms(item["text"], item["aliases"])
            scopes, words = tuple(zip(*anchors, strict=True))
            for view in discovery.VIEWS:
                left, right = item["dois"]
                clues = [payload["clues"][doi][view] for doi in (left, right)]
                prompts = [f"Instruct: {discovery.DENSE_TASK}\nQuery: {clue}" for clue in clues]
                prompts += [f"Instruct: {EXPERT_TASK}\nQuery:{word}" for word in words]
                for tag in discovery.TAGS:
                    removed = {"missing_left": (left,), "missing_right": (right,)}.get(tag, ())
                    for policy in SOURCE_POLICIES:
                        evaluated = {arm: forward(index, evidence, prompts, clues, scopes, removed, banks["ridge"],
                                                  cutoff, policy, arm)
                                     for arm in ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]}
                        if not i:
                            continue
                        for arm, result in evaluated.items():
                            key = view, tag, policy, arm
                            row = count(result, item, removed, evidence)
                            results = counts[key]
                            results.update(row)
                            results["matches_separate"] += result[-1] == evaluated["separate"][-1]
                            unique = unique_sources[key]
                            pair = tuple(item["dois"])
                            unique.setdefault(pair, set()).add(result[0])
                            if not removed:
                                pair_exact[key][item["family"]].append(row["all_owners_and_sources"])
                            timings[key].append(result[3])
    report = {}
    for key, values in counts.items():
        distinct = unique_sources[key]
        report["/".join(key)] = dict(values, p95_ms=float(np.quantile(timings[key], .95)),
                                     unique_pairs=len(distinct),
                                     unique_pair_exact=sum(answers == {pair} for pair, answers in distinct.items()),
                                     unique_pairs_with_wrong_answer=sum(bool(answers - {pair, None})
                                                                        for pair, answers in distinct.items()),
                                     both_versions_exact=sum(len(v) == 2 and all(v) for v in pair_exact[key].values()))
    discovery.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                      script_sha256=discovery.digest(Path(__file__)), prior_sha256=discovery.digest(prior),
                                      fixture_sha256=discovery.FIXTURE, projection_sha256=discovery.PROJECTION,
                                      ranker_sha256=discovery.digest(Path(discovery.__file__)), queries=len(items),
                                      indexed_sources=len(index.dois), embedding_revision=provenance["revision"],
                                      offline_bank_ms=offline_ms, cutoff=cutoff, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture_path", "glossary_path", "projection_path", "training", "embedding", "prior", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
