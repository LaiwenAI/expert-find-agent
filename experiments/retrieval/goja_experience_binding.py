"""Attach annotated experience expressions to the right target in real job ads."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARCHIVE = "a1b1408656283268470e8103b676229dd8e7e692fd361a1ed463460c32ce950c"
ENTITY_ID = "{http://www.omg.org/XMI}id"
TARGET = {"Arbeitsinhalt", "Prozess"}
RELATIONS = {"Nullrelation", "Arbeitsmittel"}
INSTRUCTION = "Identify the work activity or resource to which the stated prior experience applies."
PAD = 180
DEPTH = 5


@dataclass(frozen=True, order=True)
class Entity:
    start: int
    end: int
    kind: str


def load_ads(path, minimum_links=2):
    if digest(path) != ARCHIVE:
        raise ValueError("Expected the pinned anonymized GOJA release")
    seen = set()
    with ZipFile(path) as source:
        for name in sorted(source.namelist()):
            if not name.endswith(".xmi"):
                continue
            doc = ElementTree.fromstring(source.read(name))
            text = next(row for row in doc if row.tag.endswith("}Sofa")).attrib["sofaString"]
            normalized = " ".join(text.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            # UIMA offsets are UTF-16 code units, even for anonymized text.
            offsets = {0: 0}
            end = 0
            for i, character in enumerate(text, 1):
                end += 1 + (ord(character) > 0xffff)
                offsets[end] = i
            entities = {row.attrib[ENTITY_ID]: Entity(offsets[int(row.attrib["begin"])],
                                                       offsets[int(row.attrib["end"])], row.attrib["Entity_Type"])
                        for row in doc if row.tag.endswith("}LM_Entity")}
            linked = defaultdict(set)
            for row in doc:
                if not row.tag.endswith("}LM_Relation") or row.attrib["Relation_Type"] not in RELATIONS:
                    continue
                a, b = (entities.get(row.attrib[key]) for key in ("Governor", "Dependent"))
                if a is not None and b is not None:
                    for anchor, other in ((a, b), (b, a)):
                        if anchor.kind == "Erfahrungsniveau" and other.kind in TARGET:
                            linked[anchor].add(other)
            if len(linked) >= minimum_links:
                yield sha256(normalized.encode()).hexdigest(), text, linked


def gap(a, b):
    return max(a.start - b.end, b.start - a.end, 0)


def bounds(a, b):
    first, second = sorted((a, b))
    return first.end, second.start


def same_line(text, a, b):
    first, second = bounds(a, b)
    return "\n" not in text[first:second]


def snippet(text, entity):
    # Preserve page layout inside the local neighborhood and identify the span by offsets.
    left, right = max(0, entity.start - 55), min(len(text), entity.end + 55)
    return text[left:entity.start] + " [" + text[entity.start:entity.end] + "] " + text[entity.end:right]


def make_cases(path):
    ads = list(load_ads(path))
    for group, text, linked in ads:
        eligible = set().union(*linked.values())
        for anchor, positives in sorted(linked.items()):
            pool = sorted((b for b in eligible if b != anchor and gap(anchor, b) <= PAD),
                          key=lambda b: (gap(anchor, b), b))[:DEPTH]
            positive_pool = [b for b in pool if b in positives]
            negative_pool = [b for b in pool if b not in positives and
                             not any(gap(b, p) == 0 for p in positives)]
            if len(positive_pool) != 1 or not negative_pool:
                continue
            menu = sorted(positive_pool + negative_pool)
            yield group, text, anchor, menu, menu.index(positive_pool[0])


def encode_cases(cases, model):
    cache = {}
    for contextual in (() if model is None else (False, True)):
        query, target = set(), set()
        for _, text, anchor, menu, _ in cases:
            show = snippet if contextual else lambda s, item: s[item.start:item.end]
            query.add(f"Instruct: {INSTRUCTION}\nQuery:{show(text, anchor)}")
            target.update(show(text, candidate) for candidate in menu)
        options = sorted(query | target)
        vectors = model.encode(options)
        cache[contextual] = dict(zip(options, vectors, strict=True))

    prepared = []
    for group, text, anchor, menu, correct in cases:
        rows = []
        for candidate in menu:
            distance = gap(anchor, candidate)
            line = float(same_line(text, anchor, candidate))
            cosines = []
            for contextual, bank in cache.items():
                show = snippet if contextual else lambda s, item: s[item.start:item.end]
                query = f"Instruct: {INSTRUCTION}\nQuery:{show(text, anchor)}"
                cosines.append(float(bank[query] @ bank[show(text, candidate)]))
            rows.append([-np.log1p(distance), line, float(candidate.start >= anchor.end),
                         float(candidate.kind == "Prozess"), *cosines])
        prepared.append((group, np.array(rows), correct))
    return prepared


def fit(rows, columns):
    # Each anchor contributes total unit weight, evenly split between its positive
    # and the average of its available negatives.
    design, targets, weights = [], [], []
    for _, features, answer in rows:
        values = features[:, columns]
        labels = np.arange(len(values)) == answer
        design.extend(values)
        targets.extend(labels)
        weights.extend(np.where(labels, .5, .5 / (len(values) - 1)))
    x, y, weight = map(np.asarray, (design, targets, weights))
    center, scale = x.mean(axis=0), x.std(axis=0).clip(.001)
    z = (x - center) / scale
    coefficients = np.linalg.solve(z.T @ (z * weight[:, None]) + np.eye(z.shape[1]), z.T @ (y * weight))
    return lambda features: ((features[:, columns] - center) / scale) @ coefficients


def score(prepared):
    pockets = {"fit": [], "test": []}
    for row in prepared:
        bucket = int(row[0][:8], 16) % 10
        pockets["fit" if bucket < 6 else "test"].append(row)
    train, test = pockets["fit"], pockets["test"]
    if not train or not test:
        raise ValueError("Expected both grouped splits")
    signals = {
        "nearest": lambda x: x[:, 0],
        "line_first": lambda x: 10 * x[:, 1] + x[:, 0],
        "learned_structure": fit(train, [0, 1, 2, 3]),
    }
    if train[0][1].shape[1] > 4:
        signals.update(
            span_cosine=lambda x: x[:, 4],
            window_cosine=lambda x: x[:, 5],
            learned_span=fit(train, [0, 1, 2, 3, 4]),
            learned_context=fit(train, [0, 1, 2, 3, 4, 5]),
        )
    groups = [g for g, _, _ in test]
    outputs = {}
    for name, function in signals.items():
        outputs[name] = [int(function(features).argmax()) == answer for _, features, answer in test]
    report = {}
    for name, items in outputs.items():
        report[name] = dict(correct=sum(items), total=len(items),
                            paired_vs_nearest=clustered_paired_interval(
                                [int(a) - int(b) for a, b in zip(items, outputs["nearest"], strict=True)], groups),
                            paired_vs_learned_structure=clustered_paired_interval(
                                [int(a) - int(b) for a, b in zip(items, outputs["learned_structure"], strict=True)], groups))
    return dict(results=report, fit_anchors=len(train), fit_docs=len({r[0] for r in train}),
                test_anchors=len(test), test_docs=len(set(groups)),
                test_menu_width=dict(Counter(len(r[1]) for r in test)),
                order_invariant_signature=sha256(json.dumps(
                    sorted((group, index, *[values[index] for values in outputs.values()])
                           for index, group in enumerate(groups))).encode()).hexdigest())


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run under Slurm with a fresh result path")
    cases = list(make_cases(args.archive))
    model, revision = None, None
    if args.embedding is not None:
        if not torch.cuda.is_available():
            raise ValueError("Embedding requires Slurm CUDA")
        revision = verify_checkpoint(args.embedding)["revision"]
        torch.backends.cuda.matmul.allow_tf32 = False
        model = QwenTorchEmbedding(args.embedding)
    started = perf_counter()
    ordered = cases[::-1] if args.reverse else cases
    prepared = encode_cases(ordered, model)
    if args.reverse:
        prepared.reverse()
    result = score(prepared)
    write_json(args.output, dict(**result, archive=ARCHIVE, runner=digest(Path(__file__)),
                                revision=revision, reverse=args.reverse, elapsed=perf_counter() - started,
                                gpu=torch.cuda.get_device_name() if model is not None else None,
                                job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--embedding", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
