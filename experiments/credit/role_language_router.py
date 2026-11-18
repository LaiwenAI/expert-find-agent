"""Benchmark local normalization of human-written role descriptions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.attribution_controls import read_inputs
from experiments.credit.plos_role_inventory import ROLES
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


NISO_SHA256 = "c107b4e780218e58aec4e2a13f2d7072533962a6a73b82d23e4273b43aafd528"
METHODS = ("lexical", "dense", "full_reader", "shortlist_reader", "agreement_then_reader")
SYMBOLS = "ABCDEFGHIJKLMN"
TASK = "Match the described research task to the most applicable contributor role."
REPEATS = 3


class Router:
    def __init__(self, docs, lexical, vectors, embedder, reader, tokenizer, shortlist_size):
        self.docs = docs
        self.lexical = lexical
        self.vectors = vectors
        self.embedder = embedder
        self.reader = reader
        self.tokenizer = tokenizer
        self.shortlist_size = shortlist_size
        encoded = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in SYMBOLS]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Each choice symbol must map to one token")
        self.ids = [ids[0] for ids in encoded]

    def lexical_rank(self, description):
        hits = self.lexical.document_scores(description)
        ranks = sorted(range(len(ROLES)), key=lambda i: (-hits.get(str(i), 0), i))
        return ranks, bool(hits)

    def dense_rank(self, description):
        vector = self.embedder.encode([f"Instruct: {TASK}\nQuery: {description}"])[0]
        return np.argsort(-(self.vectors @ vector), kind="stable").tolist()

    def choose(self, description, indices, example_index):
        shift = example_index % len(indices)
        ordered = indices[shift:] + indices[:shift]
        options = "\n".join(f"{symbol}: {self.docs[i]}" for symbol, i in zip(SYMBOLS, ordered))
        prompt = (f"{TASK} Choose the single best option from the glossary.\n"
                  f"{options}\n\nResearch task: {description}\nAnswer:")
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        output = self.reader(**inputs, logits_to_keep=1, use_cache=False)
        logits = output.logits[0, -1, self.ids[:len(ordered)]]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite role scores")
        return ordered[int(logits.argmax().item())], inputs.input_ids.shape[1]

    def run(self, description, method, i):
        start = perf_counter()
        language_call = False
        options = 0
        input_tokens = 0
        tested_candidates = None
        if method == "lexical":
            index = self.lexical_rank(description)[0][0]
        elif method == "dense":
            index = self.dense_rank(description)[0]
        elif method == "full_reader":
            options = len(ROLES)
            index, input_tokens = self.choose(description, list(range(options)), i)
            language_call = True
        else:
            lexical, lexical_hit = self.lexical_rank(description)
            dense = self.dense_rank(description)
            if method == "agreement_then_reader" and lexical_hit and lexical[0] == dense[0]:
                index = lexical[0]
            else:
                candidates = (sorted(set(lexical[:self.shortlist_size] + dense[:self.shortlist_size]))
                              if method == "shortlist_reader" else list(range(len(ROLES))))
                options = len(candidates)
                tested_candidates = candidates
                index, input_tokens = self.choose(description, candidates, i)
                language_call = True
        return dict(predicted=ROLES[index], ms=(perf_counter() - start) * 1000, called_reader=language_call,
                    options=options, reader_tokens=input_tokens, candidates=tested_candidates)


def load_fixture(path, sources):
    queries, raw_papers, label_hash = read_inputs(path, sources)
    papers = defaultdict(dict)
    for row in queries:
        papers[row["doi"]][row["requested_role"]] = set(row["matching_author_ordinals"])
    full_declarations = {}
    for doi, role_queries in papers.items():
        memberships = raw_papers[doi].roles
        if not memberships or any(not roles for roles in memberships):
            raise ValueError("Compound fixture requires all author declarations")
        roles = {name: {i for i, assigned in enumerate(memberships, 1) if name in assigned} for name in ROLES}
        if any(roles[name] != expected for name, expected in role_queries.items()):
            raise ValueError("Compound fixture differs from publisher roles")
        full_declarations[doi] = roles
    return papers, full_declarations, label_hash


def interpret_compounds(rows, papers, declarations):
    examples = defaultdict(list)
    for row in rows:
        examples[row["role"]].append(row)
    counts = {name: Counter() for name in METHODS}
    for doi, gold in sorted(papers.items()):
        for role_a, role_b in combinations(sorted(gold), 2):
            expected = gold[role_a] & gold[role_b]
            controls = []
            for role in (role_a, role_b):
                key = f"{doi}:{role_a}:{role_b}:{role}"
                index = int(sha256(key.encode()).hexdigest(), 16) % len(examples[role])
                controls.append(examples[role][index])
            for method in METHODS:
                first, second = [row["arms"][method][0]["predicted"] for row in controls]
                proper = {first, second} == {role_a, role_b}
                predicted = declarations[doi][first] & declarations[doi][second]
                report = counts[method]
                report["queries"] += 1
                report["exact_role_pair"] += int(proper)
                report["exact_author_set"] += int(predicted == expected)
                report["any_unsupported_author"] += int(bool(predicted - expected))
    return counts


def aggregate(rows):
    metrics = {}
    for mode in METHODS:
        predictions = [row["arms"][mode][0] for row in rows]
        p = np.quantile([np.median([e["ms"] for e in row["arms"][mode]]) for row in rows], [.5, .95])
        result = {
            "correct": sum(row["role"] == pred["predicted"] for row, pred in zip(rows, predictions, strict=True)),
            "examples": len(rows),
            "request_p50_p95_ms": [float(q) for q in p],
            "calls": sum(pred["called_reader"] for pred in predictions),
            "mean_reader_tokens_per_request": mean(pred["reader_tokens"] for pred in predictions),
        }
        if mode == "shortlist_reader":
            result["gold_in_candidates"] = sum(ROLES.index(row["role"]) in pred["candidates"]
                                               for row, pred in zip(rows, predictions, strict=True))
        metrics[mode] = result
    return metrics


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.examples) != NISO_SHA256:
        raise ValueError("Changed NISO examples")
    data = json.loads(args.examples.read_bytes())
    if tuple(data["definitions"]) != ROLES:
        raise ValueError("Unexpected taxonomy")
    docs = [f"{role}: {data['definitions'][role]}" for role in ROLES]
    papers, declarations, fixture_hash = load_fixture(args.fixture, args.sources)
    evidence = json.loads((args.embedding / "checkpoint-provenance.json").read_bytes())
    for name, checksum in evidence["files"].items():
        if digest(args.embedding / name) != checksum:
            raise ValueError(f"Changed embedding file: {name}")
    embedding = QwenTorchEmbedding(args.embedding)
    embedding.model.config.use_cache = False
    vectors = embedding.encode(docs)
    tokenizer, model, checkpoint = load_reader(args.reader)
    with TemporaryDirectory() as folder:
        baseline = build_lexical({text: i for i, text in enumerate(docs)}, Path(folder))
    router = Router(docs, baseline, vectors, embedding, model, tokenizer, args.shortlist_size)
    rows = []
    with torch.inference_mode():
        for mode in METHODS:
            router.run("Record a research task.", mode, 0)
        for i, item in enumerate(data["examples"]):
            arms = defaultdict(list)
            for repeat in range(REPEATS):
                shift = (i + repeat) % len(METHODS)
                for mode in METHODS[shift:] + METHODS[:shift]:
                    arms[mode].append(router.run(item["text"], mode, i + args.option_offset))
            if any(len({row["predicted"] for row in repeat}) != 1 for repeat in arms.values()):
                raise ValueError("Repeated role choice changed")
            rows.append(dict(role=item["role"], ordinal=item["ordinal"], arms=arms))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "examples.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "source_doi": data["doi"], "niso_input_sha256": NISO_SHA256, "fixture_sha256": fixture_hash,
        "rows_sha256": digest(path), "checkpoint": checkpoint, "embedding_checkpoint": evidence,
        "gpu": torch.cuda.get_device_name(), "repeats": REPEATS, "example_metrics": aggregate(rows),
        "shortlist_size": args.shortlist_size, "option_offset": args.option_offset,
        "compound_metrics": interpret_compounds(rows, papers, declarations),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("benchmarks/plos-credit-pilot-v1"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--reader", type=Path, default=Path("artifacts/models/qwen3-4b-instruct-2507"))
    parser.add_argument("--option-offset", type=int, default=0)
    parser.add_argument("--shortlist-size", type=int, choices=range(1, len(ROLES) + 1), default=3)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
