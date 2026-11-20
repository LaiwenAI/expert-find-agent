"""Evaluate prose topic-and-role requests through the joint source path."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding import TASK as TOPIC_TASK
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_expert_bundles as wording
from experiments.credit import live_joint_source_request as live
from experiments.credit import role_prototypes as prototypes
from experiments.followir.core17_reranking import digest, write_json
from experiments.qasper.paired_reader import load_reader


BASE = live.BASE
ARMS = ("known_roles", "ridge", "fused_ridge", "reader", "guarded")
PATTERNS = (
    re.compile(r'^Find one person\. In research on "([^"]+)", they must have done (.*?); separately, '
               r'in research on "([^"]+)", they must have done (.*)\.$'),
    re.compile(r'^Look for one researcher who did (.*?) in work on (.*?); in addition, '
               r'who did (.*?) in work on (.*)\.$'),
)


def parse(text):
    first = PATTERNS[0].fullmatch(text)
    if first:
        first_topic, first_phrase, second_topic, second_phrase = first.groups()
    else:
        match = PATTERNS[1].fullmatch(text)
        if match is None:
            return None
        first_phrase, first_topic, second_phrase, second_topic = match.groups()
    return (first_topic, second_topic), (first_phrase, second_phrase)


def phrase_pick(held, atoms, i):
    role = atoms[i][1]
    variants = held[role]
    index = int.from_bytes(sha256(repr((atoms, i)).encode()).digest()[:4], "big") % len(variants)
    return variants[index]


def prose(atoms, held, style):
    (a, _), (b, _) = atoms
    x, y = (phrase_pick(held, atoms, i).strip().rstrip(".") for i in range(2))
    if style == "quoted":
        return f'Find one person. In research on "{a}", they must have done {x}; separately, in research on "{b}", they must have done {y}.'
    return f"Look for one researcher who did {x} in work on {a}; in addition, who did {y} in work on {b}."


def pick_ridge(encoded_tasks, bank):
    values = encoded_tasks @ bank.T
    order = np.argsort(-values, axis=1, kind="stable")
    best = [prototypes.ROLES[i] for i in order[:, 0]]
    selected = np.take_along_axis(values, order[:, :2], axis=1)
    return best, selected[:, 0] - selected[:, 1]


def resolve(index, encoder, router, threshold, text, gold, arm, i):
    start = perf_counter()
    parsed = parse(text)
    if parsed is None:
        return None, (perf_counter()-start)*1000, 0, False, False
    topics, phrases = parsed
    gold_topics = tuple(a[0] for a in gold)
    gold_roles = tuple(a[1] for a in gold)
    rank_encoder = encoder
    reader_tasks = 0
    if arm == "known_roles":
        roles = gold_roles
    elif arm == "reader":
        roles = router.read_roles(phrases, i)
        reader_tasks = 2
    elif arm == "fused_ridge":
        queries = [f"Instruct: {TOPIC_TASK}\nQuery:{t}" for t in topics]
        tasks = [f"Instruct: {prototypes.TASK}\nQuery: {p}" for p in phrases]
        vectors = encoder.encode(queries + tasks)
        roles, _ = pick_ridge(vectors[2:], router.banks["ridge"])
        rank_encoder = SimpleNamespace(encode=lambda *args, **kwargs: vectors[:2])
    else:
        roles, gaps = router.dense(phrases, "ridge")
        if arm == "guarded":
            low = [j for j, gap in enumerate(gaps) if gap < threshold]
            if low:
                predictions = router.read_roles([phrases[j] for j in low], i)
                for j, role in zip(low, predictions, strict=True):
                    roles[j] = role
                reader_tasks = len(low)
    index.encoder = rank_encoder
    correct_roles = tuple(roles) == gold_roles
    correct_topics = topics == gold_topics
    atoms = tuple(zip(topics, roles, strict=True))
    ranks = index.rank(list(topics))
    chosen, loaded = live.inspect(index, atoms, ranks, "joint")
    if len(loaded) != 3:
        raise AssertionError("Read budget changed")
    result = live.render(loaded, chosen, atoms, ranks)
    return result, (perf_counter()-start)*1000, reader_tasks, correct_topics, correct_roles


def summarize(index, encoder, router, threshold, cases, catalogue, subjects, held, offset, style):
    stats = {arm: Counter() for arm in ARMS}
    times = {arm: [] for arm in ARMS}
    for arm in ARMS:
        resolve(index, encoder, router, threshold, prose(cases[0], held, style), cases[0], arm, offset)
    for i, gold in enumerate(cases):
        text = prose(gold, held, style)
        shift = (offset + i) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            result, elapsed, calls, correct_topics, correct_roles = resolve(
                index, encoder, router, threshold, text, gold, arm, i,
            )
            value = stats[arm]
            value.update(aligned_topics=int(correct_topics), aligned_roles=int(correct_roles), reader_tasks=calls)
            matches = json.loads(result)["matches"] if result is not None else []
            value["issued"] += int(bool(matches))
            if matches:
                person, *docs = matches[0]
                role_ok = all(any(a["person"] == person and role in a["roles"]
                                  for a in catalogue.ownership[doi]["authors"])
                              for (_, role), doi in zip(gold, docs, strict=True))
                topic_ok = all(topic in subjects[doi] for (topic, _), doi in zip(gold, docs, strict=True))
                value.update(known=int(role_ok and topic_ok), wrong_declared_role=int(not role_ok),
                             topic_unresolved_with_correct_role=int(role_ok and not topic_ok))
            times[arm].append(elapsed)
    return {arm: dict(stats[arm], p95_ms=float(np.quantile(times[arm], .95))) for arm in ARMS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "glossary", "training", "embedding",
                 "reader", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--order-offset", type=int, default=0)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on Slurm CUDA")
    if digest(args.glossary) != wording.NISO_SHA256:
        raise ValueError("Changed NISO example inventory")
    checkpoint = live.verify_embedding(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    tokenizer, model, reader = load_reader(args.reader)
    glossary = json.loads(args.glossary.read_bytes())
    bank = SimpleNamespace(embedder=encoder, vectors=prototypes.encode_tasks(
        encoder, [f"{role}: {glossary['definitions'][role]}" for role in prototypes.ROLES]
    ))
    banks, _ = prototypes.train_banks(args.training, bank, glossary["examples"])
    router = wording.Router(glossary, encoder, tokenizer, model, banks, chat_reader=True)
    cal, held = wording.phrase_split(glossary)
    choice = wording.calibrate(router, cal)
    threshold = choice["winner"]["threshold"]
    plos, _ = BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    plos_subjects = BASE.publisher_topics([args.plos_xml])
    life_subjects, _, life_cases = live.elife.build_requests(life)["topics"]
    with TemporaryDirectory() as scratch:
        panels = dict(plos=(plos, plos_subjects, BASE.query_panel(BASE.annotation_postings(plos, plos_subjects))),
                      elife=(life, life_subjects, life_cases))
        results = {}
        for name, (catalogue, subjects, cases) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            results[name] = {style: summarize(index, encoder, router, threshold, cases, catalogue,
                                              subjects, held, args.order_offset, style)
                             for style in ("quoted", "unquoted")}
            print(f"Completed {name}", flush=True)
    write_json(args.output, dict(
        results=results, calibration=choice, embedding_checkpoint=checkpoint, reader_checkpoint=reader["revision"],
        projection_sha256=projection, training_sha256=digest(args.training), glossary_sha256=digest(args.glossary),
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
