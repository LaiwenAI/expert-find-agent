"""Evaluate frozen compact-reader thresholds on disjoint SciFact-Open claim and document pairs."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
import os
from pathlib import Path
import tarfile
from time import perf_counter
import unicodedata


PINS = {
    "archive": "8f4dc238b2eff422bab5c7918fcbddb6996b29d37cb887942ef6ee1a16fb3415",
    "original": "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
    "exposure": "9b57f06d84bcdb3a633f41d0c9064fa0ee2a72b76e5494bebc58b1a9cae6cc76",
}
CUTS = {"full": 1., "two_lexical_windows": -3.75}
MAX_CHARACTERS = 2500


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prepare(archive, original, exposure, output):
    if output.exists() or {key: digest(path) for key, path in
                           (("archive", archive), ("original", original), ("exposure", exposure))} != PINS:
        raise ValueError("Require pinned inputs and a new held-pair projection")
    with tarfile.open(original, "r:gz") as source:
        prior = [json.loads(line) for part in ("train", "dev")
                 for line in source.extractfile(f"data/claims_{part}.jsonl")]
    old = json.loads(exposure.read_text())
    blocked = {row["id"] for row in prior} | {row["claim_id"] for row in old["groups"]}
    blocked_text = {normalized(row["claim"]) for row in (*prior, *old["groups"])}
    documents = {str(doc) for row in prior for doc in (*row["cited_doc_ids"], *row["evidence"])} | set(old["abstracts"])
    target_docs = set()
    pairs = []
    corpus = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if member.name == "data/claims.jsonl":
                for line in source.extractfile(member):
                    row = json.loads(line)
                    if row["id"] in blocked or normalized(row["claim"]) in blocked_text:
                        continue
                    for doc, info in row["evidence"].items():
                        if doc in documents:
                            continue
                        label = info["label"].lower()
                        if label not in ("support", "contradict"):
                            raise ValueError("Unexpected evidence label")
                        target_docs.add(doc)
                        pairs.append(dict(case=row["id"], doc=doc, criterion=row["claim"],
                                          grade=label, provenance=info["provenance"]))
            if member.name == "data/corpus.jsonl":
                if not pairs:
                    raise ValueError("Expected eligible claims before the corpus")
                for line in source.extractfile(member):
                    row = json.loads(line)
                    doc = str(row["doc_id"])
                    if doc in target_docs:
                        if doc in corpus:
                            raise ValueError("Duplicate article")
                        corpus[doc] = row["abstract"]
                break
    if target_docs != set(corpus):
        raise ValueError("Missing annotated abstract")
    rows = [row | dict(sentences=corpus[row["doc"]]) for row in pairs
            if corpus[row["doc"]] and len(" ".join(corpus[row["doc"]])) <= MAX_CHARACTERS]
    with output.open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: (r["case"], r["doc"])):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(dict(cases=len(rows), claims=len({r["case"] for r in rows}),
               labels=dict(Counter(r["grade"] for r in rows)), sha256=digest(output)), flush=True)


def components(rows):
    parents = {}
    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for row in rows:
        parents[root(("claim", row["case"]))] = root(("doc", row["doc"]))
    return [root(("claim", row["case"])) for row in rows]


def compare(projected, checkpoint, output):
    import numpy as np
    import torch

    from expert_search.statistics import clustered_paired_interval
    from experiments.followir.core17_reranking import write_json
    from experiments.qasper.paired_reader import load_reader
    from experiments.scifact.compact_full_verifier import source_view
    from experiments.scifact.criterion_reader import measure
    from experiments.scifact.selective_evidence_verifier import support_margin

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh Slurm GPU run")
    rows = [json.loads(line) for line in projected.read_text().splitlines()]
    if len({(r["case"], r["doc"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluated pair")
    abstracts = {r["doc"]: r["sentences"] for r in rows}
    data = [r | {"source": " ".join(r["sentences"])} for r in rows]
    tokenizer, reader, checkpoint_info = load_reader(checkpoint)
    for arm in CUTS:
        measure(tokenizer, reader, source_view(data[:1], abstracts, arm))
    scores = {arm: [] for arm in CUTS}
    times = {arm: [] for arm in CUTS}
    for idx, row in enumerate(data):
        order = tuple(CUTS)
        for arm in order[idx % 2:] + order[:idx % 2]:
            start = perf_counter()
            values, _ = measure(tokenizer, reader, source_view([row], abstracts, arm))
            scores[arm].append(bool(support_margin(values)[0] > CUTS[arm]))
            times[arm].append((perf_counter() - start) * 1000)
    labels = np.array([r["grade"] for r in rows])
    baseline = np.asarray(scores["full"], dtype=bool)
    compact = np.asarray(scores["two_lexical_windows"], dtype=bool)
    clusters = components(rows)
    slices = {}
    for key, selector in (("all", np.ones(len(rows), dtype=bool)),
                          *((name, np.array([row["provenance"] == name for row in rows]))
                            for name in sorted({r["provenance"] for r in rows}))):
        slices[key] = {}
        for label in ("support", "contradict"):
            indices = np.flatnonzero(selector & (labels == label))
            if not indices.size:
                continue
            full, short = baseline[indices], compact[indices]
            interval = clustered_paired_interval(short.astype(int) - full.astype(int), [clusters[i] for i in indices])
            slices[key][label] = dict(
                pairs=len(indices), full_support=int(full.sum()), compact_support=int(short.sum()),
                gained=int((short & ~full).sum()), lost=int((full & ~short).sum()),
                paired_delta=interval | {"sampling_unit": "connected claim/document component"},
            )
    by_component = defaultdict(list)
    for i, group in enumerate(clusters):
        by_component[group].append(i)
    groups = tuple(by_component.values())
    rng = np.random.default_rng(1919)
    full_ms, short_ms = (np.asarray(times[name]) for name in CUTS)
    ratios = []
    for sample in rng.integers(len(groups), size=(2000, len(groups))):
        ids = np.concatenate([groups[i] for i in sample])
        ratios.append(float(np.quantile(full_ms[ids], .95) / np.quantile(short_ms[ids], .95)))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), revision=checkpoint_info["revision"],
        pairs=len(rows), claims=len({r["case"] for r in rows}), papers=len(abstracts), components=len(groups),
        projection_sha256=digest(projected), script_sha256=digest(Path(__file__)), cutoffs=CUTS,
        sliced_results=slices,
        p50_p95_ms={arm: np.quantile(values, [.5, .95]).tolist() for arm, values in times.items()},
        p95_speedup_interval=np.quantile(ratios, [.025, .975]).tolist(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(required=True)
    for mode, method, arguments in (
        ("prepare", prepare, ("archive", "original", "exposure", "output")),
        ("compare", compare, ("projected", "checkpoint", "output")),
    ):
        sub = subs.add_parser(mode)
        for name in arguments:
            sub.add_argument(name, type=Path)
        sub.set_defaults(run=method)
    args = vars(parser.parse_args())
    args.pop("run")(**args)
