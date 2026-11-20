"""Replay exact vectorized reciprocal-rank fusion with the same tie semantics."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import adaptive_role_depth as adaptive
from experiments.credit import live_joint_role_postings as postings
from experiments.credit import live_joint_source_request as live
from experiments.credit import mixed_publisher_joint as mixed
from experiments.credit import robust_role_postings as stress
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import paper_text
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


ARMS = ("python", "array_sort", "array_part")
DEPTH = 128
RRF_CONSTANT = 60


class RankFusion:
    def __init__(self, valid, lexical):
        self.lexical = lexical
        self.valid = np.asarray(valid, dtype=np.int64)
        self.lexical_ids = np.asarray([int(doc) for doc in lexical.ids], dtype=np.int64)
        if not np.array_equal(np.sort(self.lexical_ids), np.arange(len(self.lexical_ids))):
            raise ValueError("Lexical and dense document positions must match bijectively")
        self.valid_mask = np.zeros(len(self.lexical_ids), dtype=bool)
        self.valid_mask[self.valid] = True
        self.lexical_valid = self.valid_mask[self.lexical_ids]
        self.weight = 1.0 / np.arange(RRF_CONSTANT + 1, RRF_CONSTANT + len(self.valid) + 1, dtype=np.float64)

    def python_scores(self, dense, lexical):
        sparse = sorted(
            ((int(doc), score) for doc, score in lexical.items() if self.valid_mask[int(doc)]),
            key=lambda pair: (-pair[1], pair[0]),
        )
        semantic = sorted(self.valid.tolist(), key=lambda doc: (-dense[doc], doc))
        scores = Counter()
        for ranking in ([doc for doc, _ in sparse], semantic):
            for i, doc in enumerate(ranking, 1):
                scores[doc] += 1 / (RRF_CONSTANT + i)
        return sorted(scores, key=lambda doc: (-scores[doc], doc))[:DEPTH]

    def array_scores(self, dense, sparse_state, partial):
        lexical, active = sparse_state
        subset = np.flatnonzero(active & self.lexical_valid)
        ids = self.lexical_ids[subset]
        sparse = ids[np.lexsort((ids, -lexical[subset]))]
        dense_order = self.valid[np.lexsort((self.valid, -dense[self.valid]))]
        combined = np.zeros(len(dense), dtype=np.float64)
        combined[dense_order] = self.weight
        combined[sparse] += self.weight[:len(sparse)]
        eligible = self.valid
        if partial and len(eligible) > DEPTH:
            scores = combined[eligible]
            cutoff = np.partition(scores, -DEPTH)[-DEPTH]
            preferred = eligible[scores > cutoff]
            tied = np.sort(eligible[scores == cutoff])[:DEPTH - len(preferred)]
            eligible = np.concatenate((preferred, tied))
        return eligible[np.lexsort((eligible, -combined[eligible]))][:DEPTH].tolist()

    def rank(self, scores, question, arm):
        if arm == "python":
            return self.python_scores(scores, self.lexical.document_scores(question))
        return self.array_scores(scores, self.lexical.document_arrays(question), arm == "array_part")


def observations(samples):
    return {name: dict(p50_ms=float(np.quantile(ms, .50)), p95_ms=float(np.quantile(ms, .95)))
            for name, ms in samples.items()}


def public_run(index, encoder, panels, offset, *, fusion_type=RankFusion, arms=ARMS):
    ranking = fusion_type(index.valid, index.catalogue.lexical)
    role_view, _ = stress.project(postings.RolePostings(index.catalogue), index.catalogue, 100, 0, "")
    elapsed = {name: [] for name in arms}
    rank_region = {name: [] for name in arms}
    issued = Counter()
    known = Counter()
    total = 0
    for panel, (queries, subjects) in panels.items():
        for conditions in queries:
            payload = live.request(conditions)
            topics = [topic for topic, _ in conditions]
            found = {}
            shift = (total + offset) % len(arms)
            for arm in arms[shift:] + arms[:shift]:
                start = perf_counter_ns()
                request = json.loads(payload)
                atoms = tuple((item["topic"], item["role"]) for item in request["requirements"])
                vectors = encoder.encode(topics, query=True)
                scores = vectors @ index.embeddings.T
                score_time = perf_counter_ns()
                orders = [ranking.rank(dense, topic, arm) for topic, dense in zip(topics, scores, strict=True)]
                rank_region[arm].append((perf_counter_ns() - score_time)/1e6)
                ranks = tuple({index.ids[doc]: j for j, doc in enumerate(ranked, 1)} for ranked in orders)
                proof = adaptive.fixed(index, atoms, ranks, role_view)
                wire = json.dumps({"matches": [proof] if proof else []}, separators=(",", ":"))
                elapsed[arm].append((perf_counter_ns() - start)/1e6)
                found[arm] = orders, wire
            if any(found[arm] != found[arms[0]] for arm in arms):
                raise AssertionError("Public ranking, expert identity or byte output changed")
            # Also keep the reference aligned with the original live entry point.
            original = index.rank(topics, depth=DEPTH)
            if [list(facet) for facet in original] != [[index.ids[i] for i in side] for side in found[arms[0]][0]]:
                raise AssertionError("Benchmark reference does not reproduce the original ranker")
            proof = json.loads(found[arms[0]][1])["matches"]
            issued[panel] += bool(proof)
            known[panel] += bool(proof) and all(
                topic in subjects.get(doc, ()) for topic, doc in zip(topics, proof[0][1:], strict=True)
            )
            total += 1
    return dict(queries=total, exact_rank_and_response=total, known=known, issued=issued,
                bm25_order_and_fusion=observations(rank_region), request=observations(elapsed))


def litsearch_run(projection, cache, encoder, revision, scratch, offset, *, fusion_type=RankFusion, arms=ARMS):
    with gzip.open(projection, "rb") as stream:
        dataset = json.load(stream)
    documents = dataset["papers"]
    queries = [q["query"] for q in dataset["questions"] if q["query_set"].startswith("manual_")]
    matrix = {}
    for position, row in enumerate(documents):
        text = paper_text(row)
        while text in matrix:
            text += "\n"
        matrix[text] = position
    scratch.mkdir()
    lexical = build_lexical(matrix, scratch)
    with np.load(cache, allow_pickle=False) as frozen:
        if (str(frozen["projection_sha256"]) != digest(projection)
            or frozen["corpus_ids"].tolist() != [row["corpusid"] for row in documents]
            or json.loads(str(frozen["provenance"]))["revision"] != revision):
            raise ValueError("Changed LitSearch vector cache or model")
        dense = torch.from_numpy(frozen["vectors"]).to("cuda")
    vectors = encoder.encode(queries, query=True)
    fusion = fusion_type(range(len(documents)), lexical)
    elapsed = {name: [] for name in arms}
    for i, (text, query) in enumerate(zip(queries, vectors, strict=True)):
        scores = (dense @ torch.from_numpy(query).to("cuda")).cpu().numpy()
        shift = (i + offset) % len(arms)
        answers = {}
        for arm in arms[shift:] + arms[:shift]:
            start = perf_counter_ns()
            answers[arm] = fusion.rank(scores, text, arm)
            elapsed[arm].append((perf_counter_ns() - start)/1e6)
        if any(answers[arm] != answers[arms[0]] for arm in arms):
            raise AssertionError("LitSearch exact fused order changed")
    return dict(queries=len(queries), sources=len(documents), ordered_matches=len(queries),
                bm25_order_and_fusion=observations(elapsed),
                projection_sha256=digest(projection), cache_sha256=digest(cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference",
                 "litsearch", "litsearch_vectors", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    elife, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"]
    ):
        raise ValueError("Frozen publisher inputs differ from reference")
    if mixed.raw_identifiers(args.plos_xml) & mixed.raw_identifiers(args.elife_xml):
        raise ValueError("A mixed catalog with shared identities requires a linked fixture")
    plos_topics = base.publisher_topics([args.plos_xml])
    elife_topics, _, elife_queries = live.elife.build_requests(elife)["topics"]
    panels = dict(plos=(base.query_panel(base.annotation_postings(plos, plos_topics)), plos_topics),
                  elife=(elife_queries, elife_topics))
    model = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        scratch = Path(temp)
        index = live.LocalIndex(mixed.merge(plos, elife), model, scratch / "publisher")
        public = public_run(index, model, panels, args.order_offset)
        print("Completed live publisher replay", flush=True)
        scale = litsearch_run(
            args.litsearch, args.litsearch_vectors, model, checkpoint, scratch / "litsearch", args.order_offset,
        )
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint, order_offset=args.order_offset,
        publisher_articles=len(index.ids), depth=DEPTH, script_sha256=digest(Path(__file__)),
        publisher=public, litsearch=scale,
    ))


if __name__ == "__main__":
    main()
