"""Avoid full dense sorting using exact reciprocal-rank bounds."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.credit import mixed_publisher_joint as mixed
from experiments.credit import vector_hybrid_rrf as previous
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("array_part", "packed_full", "bounded")
DENSE_PREFIX = 512
SIGN = np.uint32(0x80000000)
DEPTH = previous.DEPTH


def dense_keys(values, positions):
    """Pack float32 scores and unique integer tie-breaks into sortable keys."""
    if values.dtype != np.float32 or positions.dtype != np.int64:
        raise ValueError("Expected float32 scores and int64 positions")
    bits = values.view(np.uint32).copy()
    bits[values == 0] = 0  # Signed zeros tie, as in the score sort.
    ascending = np.where(bits & SIGN, ~bits, bits ^ SIGN)
    return (~ascending).astype(np.uint64) << np.uint64(32) | positions.astype(np.uint64)


def boundary_select(scores, ids):
    if len(ids) <= DEPTH:
        return ids[np.lexsort((ids, -scores))].tolist()
    cutoff = np.partition(scores, -DEPTH)[-DEPTH]
    preferred = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    slots = DEPTH - len(preferred)
    selected = np.concatenate((preferred, tied[np.argsort(ids[tied])[:slots]]))
    return ids[selected[np.lexsort((ids[selected], -scores[selected]))]].tolist()


class BoundedFusion(previous.RankFusion):
    def __init__(self, valid, lexical, *, work=None):
        super().__init__(valid, lexical)
        if len(self.lexical_ids) > 2**32:
            raise ValueError("Packed source positions must fit in 32 bits")
        self.dense_positions = np.empty(len(self.lexical_ids), dtype=np.int64)
        self.dense_positions[self.valid] = np.arange(len(self.valid))
        self.work = work

    def sparse_ranks(self, sparse_state):
        scores, active = sparse_state
        subset = np.flatnonzero(active & self.lexical_valid)
        ids = self.lexical_ids[subset]
        return ids[np.lexsort((ids, -scores[subset]))]

    def packed_full(self, keys, sparse):
        combined = np.zeros(len(self.lexical_ids), dtype=np.float64)
        combined[self.valid[np.argsort(keys)]] = self.weight
        combined[sparse] += self.weight[:len(sparse)]
        return boundary_select(combined[self.valid], self.valid)

    def bounded(self, keys, sparse):
        size = min(DENSE_PREFIX, len(keys))
        if size == len(keys):
            prefix = np.argsort(keys)
        else:
            top = np.argpartition(keys, size-1)[:size]
            prefix = top[np.argsort(keys[top])]
        prefix_ids = self.valid[prefix]
        weights = np.zeros(len(self.lexical_ids), dtype=np.float64)
        weights[sparse] = self.weight[:len(sparse)]
        prefix_scores = self.weight[:size] + weights[prefix_ids]

        if size == len(keys):
            outside_ids = np.empty(0, dtype=np.int64)
        else:
            cutoff = np.partition(prefix_scores, -DEPTH)[-DEPTH]
            # An unseen dense hit can have at best rank size + 1.
            possible = sparse[(weights[sparse] + self.weight[size]) >= cutoff]
            outside_ids = np.setdiff1d(possible, prefix_ids, assume_unique=True)
        if self.work is not None:
            self.work.append((len(sparse), len(outside_ids), size))
        if len(outside_ids) == 0:
            return boundary_select(prefix_scores, prefix_ids)

        outside_keys = keys[self.dense_positions[outside_ids]]
        order = np.argsort(outside_keys)
        # Insertion buckets count every eligible key less than or equal
        # to each retained key, giving its exact one-based dense rank.
        bins = np.searchsorted(outside_keys[order], keys, side="left")
        counts = np.bincount(bins, minlength=len(outside_keys)+1)
        ordinal = np.cumsum(counts)[:-1]
        sorted_outside = outside_ids[order]
        outer_scores = self.weight[ordinal-1] + weights[sorted_outside]
        return boundary_select(np.concatenate((prefix_scores, outer_scores)),
                               np.concatenate((prefix_ids, sorted_outside)))

    def rank(self, scores, question, arm):
        sparse_state = self.lexical.document_arrays(question)
        if arm == "array_part":
            return self.array_scores(scores, sparse_state, partial=True)
        sparse = self.sparse_ranks(sparse_state)
        keys = dense_keys(scores[self.valid], self.valid)
        if arm == "packed_full":
            return self.packed_full(keys, sparse)
        return self.bounded(keys, sparse)


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
    work = []
    with TemporaryDirectory(dir=args.output.parent) as root:
        scratch = Path(root)
        index = live.LocalIndex(mixed.merge(plos, elife), model, scratch / "publisher")
        public = previous.public_run(
            index, model, panels, args.order_offset, fusion_type=BoundedFusion, arms=ARMS,
        )
        print("Completed live publisher replay", flush=True)
        scale = previous.litsearch_run(
            args.litsearch, args.litsearch_vectors, model, checkpoint, scratch / "litsearch", args.order_offset,
            fusion_type=partial(BoundedFusion, work=work), arms=ARMS,
        )
    if len(work) != scale["queries"]:
        raise AssertionError("Incomplete selective-rank counts")
    rows = np.asarray(work)
    scale["bounded_work"] = {
        name: dict(p50=float(np.quantile(rows[:, i], .50)), p95=float(np.quantile(rows[:, i], .95)),
                   maximum=int(rows[:, i].max()))
        for i, name in enumerate(("sparse_active", "outside_exact_dense_ranks", "dense_prefix_sorted"))
    }
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        order_offset=args.order_offset, publisher_articles=len(index.ids), depth=DEPTH,
        dense_prefix=DENSE_PREFIX, script_sha256=digest(Path(__file__)),
        previous_ranker_sha256=digest(Path(previous.__file__)), publisher=public, litsearch=scale,
    ))


if __name__ == "__main__":
    main()
