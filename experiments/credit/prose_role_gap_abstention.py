"""Test a learned role-gap abstention rule without another model call."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import prose_joint_source_request as prose
from experiments.followir.core17_reranking import digest, write_json


BASE = prose.BASE
LIVE = prose.live


def calibrate(encoder, bank, atomic):
    features = prose.prototypes.encode_tasks(encoder, [phrase for phrase, _ in atomic])
    predicted, margins = prose.pick_ridge(features, bank)
    correct = np.array([role == target for role, (_, target) in zip(predicted, atomic, strict=True)])
    choices = sorted({0., *map(float, margins)})
    for threshold in choices:
        accepted = margins >= threshold
        issued = int(accepted.sum())
        right = int((accepted & correct).sum())
        if issued >= 5 and (issued - right) / issued <= .20:
            return dict(roles=len(atomic), issued=issued, correct=right, cutoff=threshold)
    return dict(roles=len(atomic), issued=0, correct=0, cutoff=None)


def evaluate(index, encoder, bank, threshold, subjects, cases, held):
    total = {name: Counter() for name in ("unfiltered", "calibrated")}
    for atoms in cases:
        topics, descriptions = prose.parse(prose.prose(atoms, held, "unquoted"))
        vectors = encoder.encode([f"Instruct: {prose.TOPIC_TASK}\nQuery:{t}" for t in topics] +
                                 [f"Instruct: {prose.prototypes.TASK}\nQuery: {phrase}" for phrase in descriptions])
        roles, margins = prose.pick_ridge(vectors[2:], bank)
        index.encoder = SimpleNamespace(encode=lambda *args, **kwargs: vectors[:2])
        predicted = tuple(zip(topics, roles, strict=True))
        ranks = index.rank(list(topics))
        chosen, loaded = LIVE.inspect(index, predicted, ranks, "joint")
        matches = json.loads(LIVE.render(loaded, chosen, predicted, ranks))["matches"]
        if not matches:
            continue
        person, *docs = matches[0]
        role_ok = all(any(a["person"] == person and role in a["roles"] for a in loaded[doi]["authors"])
                      for (_, role), doi in zip(atoms, docs, strict=True))
        topic_ok = all(topic in subjects[doi] for (topic, _), doi in zip(atoms, docs, strict=True))
        for name, permit in (("unfiltered", True), ("calibrated", threshold is not None and min(margins) >= threshold)):
            if permit:
                total[name].update(issued=1, known=int(role_ok and topic_ok), wrong_role=int(not role_ok),
                                   topic_unresolved=int(role_ok and not topic_ok))
    return {name: dict(values) for name, values in total.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "glossary", "training", "embedding", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output on Slurm")
    if digest(args.glossary) != prose.wording.NISO_SHA256:
        raise ValueError("Changed NISO example inventory")
    checkpoint = LIVE.verify_embedding(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    glossary = json.loads(args.glossary.read_bytes())
    bank = SimpleNamespace(embedder=encoder, vectors=prose.prototypes.encode_tasks(
        encoder, [f"{role}: {glossary['definitions'][role]}" for role in prose.prototypes.ROLES],
    ))
    banks, _ = prose.prototypes.train_banks(args.training, bank, glossary["examples"])
    calibration, held = prose.wording.phrase_split(glossary)
    cutoff = calibrate(encoder, banks["ridge"], calibration)
    plos, _ = BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = LIVE.elife.load_catalogue(args.elife_xml, args.manifest)
    subjects = BASE.publisher_topics([args.plos_xml])
    life_subjects, _, life_queries = LIVE.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, subjects, BASE.query_panel(BASE.annotation_postings(plos, subjects))),
                  elife=(life, life_subjects, life_queries))
    output = {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, tags, cases) in panels.items():
            index = LIVE.LocalIndex(catalogue, encoder, Path(scratch) / name)
            output[name] = dict(queries=len(cases), results=evaluate(
                index, encoder, banks["ridge"], cutoff["cutoff"], tags, cases, held,
            ))
    write_json(args.output, dict(calibration=cutoff, results=output, checkpoint=checkpoint,
                                 elife_projection=projection, script_sha256=digest(Path(__file__))))


if __name__ == "__main__":
    main()
