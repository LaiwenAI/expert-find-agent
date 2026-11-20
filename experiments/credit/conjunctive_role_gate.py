"""Test selective same-owner search on matched two-activity requests."""

import argparse
from collections import defaultdict
from itertools import combinations, product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import fresh_role_search as search
from experiments.credit import role_vocabulary_gate as gate
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json

MODEL = "single_clause_cross_entropy"
REPLICATES = 3
MAX_PER_PAIR = 3
ARMS = ("oracle", "article_union", "point", "strict_margin", "cert80", "cert100", "reader")


def build_cases(panel, bank):
    texts, declarations, queries, _ = panel
    phrases = defaultdict(list)
    for target, topic, role, phrase in queries:
        phrases[target, topic, role].append(phrase)
    grouped = defaultdict(lambda: {False: [], True: []})
    for target, topic in sorted({(q[0], q[1]) for q in queries}):
        roles = [role for role in bank if (target, topic, role) in phrases]
        for pair in combinations(sorted(roles), 2):
            truth = tuple(i for i, roles in enumerate(declarations[target]) if set(pair) <= set(roles))
            grouped[pair][bool(truth)].append((target, topic, truth))
    cases = []
    for pair, strata in grouped.items():
        n = min(MAX_PER_PAIR, len(strata[False]), len(strata[True]))
        for entries in strata.values():
            for target, topic, truth in entries[:n]:
                options = [phrases[target, topic, role] for role in pair]
                for phrase_pair in zip(*options, strict=True):
                    cases.append((target, topic, pair, phrase_pair, truth))
    if not cases:
        raise ValueError("Missing paired same-author and split-author cases")
    return texts, declarations, cases


def masks(labels):
    index = [0] * len(ROLES)
    for owner, roles in enumerate(labels):
        for role in roles:
            index[search.ROLE_IDS[role]] |= 1 << owner
    return index


def roles_in_scope(scores, deficit):
    return tuple(int(i) for i in np.flatnonzero(scores.max() - scores <= deficit))


def qualified(index, roles):
    result = index[roles[0]]
    for role in roles[1:]:
        result &= index[role]
    return result


def authors(bits):
    return tuple(i for i in range(bits.bit_length()) if bits & (1 << i))


def certify(index, left, right):
    interpretations = [qualified(index, (a, b)) for a, b in product(left, right) if a != b]
    if not interpretations:
        return "unresolved", ()
    common = interpretations[0]
    for bits in interpretations[1:]:
        common &= bits
    if common:
        return "recommend", authors(common)
    if any(interpretations):
        return "unresolved", ()
    return "no_match", ()


def decide(result):
    return ("recommend", authors(result)) if result else ("no_match", ())


def operate(arm, index, labels, wanted, scores, cutoff, deficits):
    if arm == "oracle":
        return decide(qualified(index, wanted))
    chosen = [gate.top_role(row) for row in scores]
    roles = [role for role, _ in chosen]
    if arm in deficits:
        sets = [roles_in_scope(row, deficits[arm]) for row in scores]
        return certify(index, *sets)
    if roles[0] == roles[1] or any(margin <= 0 for _, margin in chosen):
        return "unresolved", ()
    if arm == "article_union":
        matches = all(index[role] for role in roles)
        return ("recommend", tuple(range(labels))) if matches else ("no_match", ())
    if arm == "strict_margin" and any(margin <= cutoff for _, margin in chosen):
        return "unresolved", ()
    return decide(qualified(index, roles))


def outcome(status, proposed, source, target, truth):
    correct = set(proposed) & set(truth) if source == target else set()
    return dict(
        exact=int(source == target and status != "unresolved" and proposed == truth),
        good_recommendation=int(bool(proposed) and len(correct) == len(proposed)),
        any_wrong=int(len(correct) != len(proposed)),
        no_match_correct=int(source == target and status == "no_match" and not truth),
        no_match_false=int(status == "no_match" and (source != target or bool(truth))),
        unresolved=int(status == "unresolved"),
        retrieved=int(source == target),
    )


def summarize(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def report(cases, rows, times):
    groups = [case[0] for case in cases]
    contrasts = {}
    for arm in ("cert80", "cert100", "strict_margin", "reader"):
        contrasts[arm] = {key: clustered_paired_interval(
            [a[key] - b[key] for a, b in zip(rows[arm], rows["point"], strict=True)], groups,
        ) for key in ("exact", "any_wrong", "no_match_false")}
    return dict(
        queries=len(cases), papers=len(set(groups)), role_pairs=len({case[2] for case in cases}),
        positive=sum(bool(case[-1]) for case in cases), negative=sum(not case[-1] for case in cases),
        arms={arm: dict(
            all=summarize(values),
            no_qualified_person=summarize([row for row, case in zip(values, cases, strict=True) if not case[-1]]),
            qualified_person_exists=summarize([row for row, case in zip(values, cases, strict=True) if case[-1]]),
            p95_ms=float(np.quantile(times[arm], .95)),
        ) for arm, values in rows.items()},
        contrasts_vs_point=contrasts,
    )


def run(args):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.factored_article_role_verification import ROLE_TASK
    from experiments.credit.factorial_role_training import fit
    from experiments.credit.train_role_decision import encode
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a new result file")
    prior = json.loads(args.head_reference.read_bytes())
    reference = json.loads(args.gate_reference.read_bytes())
    if digest(args.training) != prior["training_sha256"]:
        raise ValueError("Changed training inputs")
    for name in ("dev_manifest", "dev_wording", "test_manifest", "test_wording"):
        if digest(getattr(args, name)) != reference["fixtures"][name]:
            raise ValueError(f"Changed source fixture: {name}")
    cutoff = reference["limits"][MODEL]
    if cutoff is None:
        raise ValueError("No frozen margin threshold")
    encoder = QwenTorchEmbedding(args.embedding)
    reader = Reranker(args.reranker)
    if prior["checkpoints"] != dict(embedding=verify_checkpoint(args.embedding)["revision"],
                                    reader=reader.provenance["revision"]):
        raise ValueError("Changed checkpoints")
    weights, _ = fit(encoder, args.training)

    def head(phrases):
        with torch.inference_mode():
            return (encode(encoder, list(phrases)) @ weights[MODEL].T).cpu().numpy()

    def read(phrases):
        values, truncated = reader.score_pairs([(p, role) for p in phrases for role in ROLES], task=ROLE_TASK)
        if truncated or not np.isfinite(values).all():
            raise ValueError("Invalid batched reader output")
        return np.asarray(values).reshape(len(phrases), len(ROLES))

    dev = json.loads(args.dev_wording.read_bytes())["activities"]
    dev_scores = head(list(dev.values()))
    misses = [max(scores) - scores[search.ROLE_IDS[role]] for scores, role in zip(dev_scores, dev, strict=True)]
    deficits = {f"cert{q}": float(np.quantile(misses, q / 100, method="higher")) for q in (80, 100)}
    held = gate.build_panel(args.test_xml, args.test_manifest, args.test_wording)
    held_bank = json.loads(args.test_wording.read_bytes())["activities"]
    texts, declarations, cases = build_cases(held, held_bank)
    author_indexes = [masks(roles) for roles in declarations]
    corpus = encoder.encode(list(texts))
    rows = {arm: [] for arm in ARMS}
    times = {arm: [] for arm in ARMS}
    coverage = {arm: 0 for arm in deficits}
    for i, (target, topic, wanted, phrases, truth) in enumerate([cases[0], *cases]):
        wanted_ids = [search.ROLE_IDS[role] for role in wanted]
        decisions = {}
        for repeat in range(REPLICATES):
            start = perf_counter()
            vector = encoder.encode([topic], query=True)[0]
            source = int(np.argmax(corpus @ vector))
            topic_ms = (perf_counter() - start) * 1000
            for model, scorer in (("head", head), ("reader", read)):
                start = perf_counter()
                scores = scorer(phrases)
                inference_ms = (perf_counter() - start) * 1000
                arms = ("reader",) if model == "reader" else ARMS[:-1]
                for arm in arms:
                    start = perf_counter()
                    status, found = operate(arm, author_indexes[source], len(declarations[source]),
                                            wanted_ids, scores, cutoff, deficits)
                    logic_ms = (perf_counter() - start) * 1000
                    measurement = outcome(status, found, source, target, truth)
                    if arm in decisions and decisions[arm] != measurement:
                        raise ValueError("Unstable replay outcome")
                    decisions[arm] = measurement
                    if i:
                        times[arm].append(topic_ms + (0 if arm == "oracle" else inference_ms) + logic_ms)
                if i and not repeat and model == "head":
                    for arm, deficit in deficits.items():
                        coverage[arm] += all(role in roles_in_scope(score, deficit)
                                             for score, role in zip(scores, wanted_ids, strict=True))
        if i:
            for arm in ARMS:
                rows[arm].append(decisions[arm])
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        fixtures={name: digest(getattr(args, name)) for name in
                  ("test_manifest", "test_wording", "head_reference", "gate_reference")},
        dev_phrase_count=len(dev), margin=cutoff, deficits=deficits, role_pair_coverage=coverage,
        latency_repeats=REPLICATES,
        **report(cases, rows, times),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev_manifest", "dev_wording", "test_xml", "test_manifest", "test_wording",
                 "training", "head_reference", "gate_reference", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
