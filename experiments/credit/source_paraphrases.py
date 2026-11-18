"""Evaluate source retrieval and abstention on paired, model-authored paper clues."""

import argparse
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.credit.conjunctive_source_retrieval import sentences
from experiments.credit.source_confirmation import accepts, measure, notice_kind
from experiments.followir.core17_reranking import digest, write_json


QUERY_SHA256 = "a91b718dca75e73471a7e17e4fa23c5f76b5310fae473e80f5b7db214c0f2158"
METHODS = ("bm25", "ordered", "dense", "reranked")
GATE_METHODS = ("agreement", "transfer", "one_reader")
STATES = ("present", "withheld")
TASK = "Identify the particular research paper described by the clue. Require the specific combination of contributions, rather than general topical relevance."


def validate(queries, fixture, catalog):
    if digest(queries) != QUERY_SHA256 or digest(fixture) != catalog["external_sha256"]:
        raise ValueError("Changed benchmark inputs")
    rows = json.loads(queries.read_bytes())
    manifest = json.loads(fixture.read_bytes())
    test = {row["doi"]: row["source_sha256"] for row in manifest["sources"]}
    records = catalog["records"]
    catalog_hashes = {row["doi"]: row["source_sha256"] for row in manifest["inventory"]}
    if any(records[doi]["source_sha256"] != expected for doi, expected in catalog_hashes.items()):
        raise ValueError("Changed publisher inputs")
    if {row["doi"] for row in rows if row["split"] == "test"} != set(test):
        raise ValueError("Expected all external sources in the test split")
    if len({row["doi"] for row in rows}) != len(rows):
        raise ValueError("Calibration and test queries must have distinct sources")
    for row in rows:
        paper = records[row["doi"]]
        if row["split"] not in {"calibration", "test"} or re.search(r'["“”\n]', row["query"]):
            raise ValueError("Expected one free-form, unquoted clue per source")
        tokens = re.findall(r"\w+", row["query"].casefold())
        text = f"{paper['title']} {paper['abstract']}".casefold()
        target = " " + " ".join(re.findall(r"\w+", text)) + " "
        if any(" " + " ".join(tokens[i:i + 8]) + " " in target for i in range(len(tokens) - 7)):
            raise ValueError(f"Copied eight-word span for {row['doi']}")
    return rows


def observation(scored):
    if not scored:
        return None, 0., 0.
    doi, score = scored[0]
    return doi, float(score), float(score - scored[1][1]) if len(scored) > 1 else 0.


class Search:
    def __init__(self, records, embedding, reranking, head_path=None):
        import torch

        from expert_search.qwen_embedding_torch import QwenTorchEmbedding
        from experiments.followir.core17_verifier import Reranker
        from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
        from experiments.retrieval.lrbench_frozen_baselines import build_lexical

        self.ids = sorted(records)
        self.docs = {doi: paper_text(records[doi]) for doi in self.ids}
        self.positions = {doi: i for i, doi in enumerate(self.ids)}
        self.segments = {
            doi: [re.findall(r"\w+", segment.casefold()) for segment in
                  [record["title"], *sentences(record["abstract"])]]
            for doi, record in records.items()
        }
        self.notices = {doi: notice_kind(r["title"]) for doi, r in records.items()}
        self.embedding_provenance = verify_checkpoint(embedding)
        self.head = None
        if head_path is not None:
            head = json.loads(head_path.read_bytes())
            if head["embedding"] != self.embedding_provenance:
                raise ValueError("Transferred head and embedding checkpoints must match")
            self.head = head["fitted"]
            self.tokens = {
                doi: tuple(set(re.findall(r"\w+", text.casefold())) for text in
                           (record["title"], record["abstract"]))
                for doi, record in records.items()
            }
        torch.backends.cuda.matmul.allow_tf32 = False
        start = perf_counter()
        with TemporaryDirectory() as folder:
            self.lexical = build_lexical(
                {f"{self.docs[doi]}{' ' * i}": i for i, doi in enumerate(self.ids)}, Path(folder)
            )
        self.lexical_build_ms = (perf_counter() - start) * 1000
        self.encoder = QwenTorchEmbedding(embedding)
        self.encoder.model.config.use_cache = False
        self.reranker = Reranker(reranking)
        start = perf_counter()
        self.vectors = self.encoder.encode([self.docs[doi] for doi in self.ids])
        self.dense_build_ms = (perf_counter() - start) * 1000

    def transfer(self, query, ranked):
        from experiments.litsearch.learned_confirmation import confirmation_features

        pool = [doi for doi, _ in ranked]
        cosine = np.asarray([value for _, value in ranked], dtype=np.float32)
        lexical, _ = self.lexical.document_arrays(query)
        sparse = lexical[[self.positions[doi] for doi in pool]]
        terms = set(re.findall(r"\w+", query.casefold()))
        title, abstract = zip(*[
            tuple(len(terms & tokens) / max(1, len(terms)) for tokens in self.tokens[doi])
            for doi in pool
        ], strict=True)
        features = confirmation_features(cosine, sparse, title, abstract)
        scores = ((features - self.head["center"]) / self.head["scale"]) @ self.head["coefficients"]
        return observation(sorted(zip(pool, scores + self.head["bias"], strict=True),
                                  key=lambda row: (-row[1], row[0])))

    def run(self, query, hidden, method):
        start = perf_counter()
        if method in {"bm25", "ordered"}:
            scores = self.lexical.document_scores(query)
            ranked = sorted(
                [(self.ids[int(i)], float(score)) for i, score in scores.items()
                 if self.ids[int(i)] != hidden], key=lambda row: (-row[1], row[0])
            )
            if method == "ordered":
                query_kind = notice_kind(query)
                ranked = [row for row in ranked if self.notices[row[0]] == query_kind]
        else:
            vector = self.encoder.encode([query], query=True)[0]
            ranked = sorted(
                [(doi, float(score)) for doi, score in zip(self.ids, self.vectors @ vector, strict=True)
                 if doi != hidden], key=lambda row: (-row[1], row[0])
            )
        pool = [doi for doi, _ in ranked[:4]]
        truncated = 0
        if method == "ordered":
            result = measure(re.findall(r"\w+", query.casefold()), pool, self.segments, "ordered")
        elif method == "reranked":
            scores, truncated = self.reranker.score(query, self.docs, pool, task=TASK)
            result = observation(sorted(zip(pool, scores, strict=True), key=lambda row: (-row[1], row[0])))
        elif method == "one_reader":
            scores, truncated = self.reranker.score(query, self.docs, pool[:1], task=TASK)
            result = pool[0], scores[0], observation(ranked)[2]
        elif method == "transfer":
            result = self.transfer(query, ranked[:4])
        elif method == "agreement":
            scores = self.lexical.document_scores(query)
            best = min((i for i in scores if self.ids[int(i)] != hidden),
                       key=lambda i: (-scores[i], self.ids[int(i)]), default=None)
            agrees = best is not None and self.ids[int(best)] == pool[0]
            result = observation(ranked) if agrees else (None, 0., 0.)
        else:
            result = observation(ranked)
        return result, pool, (perf_counter() - start) * 1000, truncated


def run(args):
    import torch

    from experiments.credit.search_role_clauses import SOURCE_POLICY
    from experiments.litsearch.source_withholding import calibrate

    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    catalog = json.loads(args.catalog.read_bytes())
    queries = validate(args.queries, args.fixture, catalog)
    all_methods = METHODS + GATE_METHODS if args.transfer_head is not None else METHODS
    search = Search(catalog["records"], args.embedding, args.reranking, args.transfer_head)
    for method in all_methods:
        search.run("Which study diagnosed electrical equipment faults?", None, method)

    rows = []
    for i, query in enumerate(queries):
        views = {state: {} for state in STATES}
        measurements = {state: {} for state in STATES}
        offset = i % len(all_methods)
        methods = all_methods[offset:] + all_methods[:offset]
        for state in STATES[::1 if i % 2 == 0 else -1]:
            hidden = query["doi"] if state == "withheld" else None
            for method in methods:
                result, pool, ms, truncated = search.run(query["query"], hidden, method)
                if hidden is not None and (hidden in pool or hidden == result[0]):
                    raise ValueError("Masked source leaked into an observation")
                views[state][method] = result
                measurements[state][method] = dict(pool=pool, ms=ms, truncated=truncated)
        rows.append(dict(**query, targets=[query["doi"]], views=views, measurements=measurements))

    calibration = [row for row in rows if row["split"] == "calibration"]
    test = [row for row in rows if row["split"] == "test"]
    learned = {method: calibrate(calibration, method, 0.)
               for method in all_methods if method not in {"bm25", "ordered"}}
    policies = {"bm25": None, "ordered": SOURCE_POLICY,
                **{method: value["policy"] for method, value in learned.items()}}
    summary = {}
    for method, policy in policies.items():
        summary[method] = {}
        for state in STATES:
            observed = [row["views"][state][method] for row in test]
            measurements = [row["measurements"][state][method] for row in test]
            selected = [
                result[0] if method == "bm25" or accepts(result, policy) else None for result in observed
            ]
            summary[method][state] = dict(
                queries=len(test),
                pool_hits=sum(row["doi"] in m["pool"] for row, m in zip(test, measurements, strict=True)),
                top_hits=sum(row["doi"] == result[0] for row, result in zip(test, observed, strict=True)),
                exact=sum(row["doi"] == result for row, result in zip(test, selected, strict=True)),
                wrong=sum(result is not None and result != row["doi"] for row, result in zip(test, selected, strict=True)),
                abstained=selected.count(None),
                p95_ms=float(np.quantile([m["ms"] for m in measurements], .95)),
                truncated=sum(m["truncated"] for m in measurements),
            )
    write_json(args.output, dict(
        inputs={name: digest(getattr(args, name)) for name in ("catalog", "fixture", "queries")},
        corpus_size=len(search.ids), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        embedding=search.embedding_provenance, reranking=search.reranker.provenance,
        lexical_build_ms=search.lexical_build_ms, dense_build_ms=search.dense_build_ms,
        policies=policies, calibration=learned, summary=summary, rows=rows,
        transfer_head_sha256=digest(args.transfer_head) if args.transfer_head is not None else None,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "fixture", "queries", "embedding", "reranking", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--transfer-head", type=Path)
    run(parser.parse_args())
