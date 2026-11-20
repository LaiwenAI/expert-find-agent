"""Verify the best multi-qualification proof under a fixed topic reader."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.credit.live_joint_role_postings import RolePostings
from experiments.credit.mixed_publisher_joint import merge, raw_identifiers
from experiments.credit.multi_qualification_proofs import (
    SHAPES, distinct, proof_key, source_postings, verified, witness_queries,
)
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("all_edges", "static9", "adaptive9", "certificate", "prefetch9")
BATCH = 12
WIDTH = 9


def feasible(postings, ranks, atoms, conflicts):
    candidates = source_postings(postings, ranks, atoms)
    return sorted(
        ((person, docs) for person, choices in candidates.items()
         for docs in product(*choices) if distinct(docs, conflicts)),
        key=lambda p: proof_key(*p, ranks),
    )


def edges(proof):
    return tuple(enumerate(proof[1]))


def frontier(proofs, observed, width):
    missing = {}
    for proof in proofs:
        parts = edges(proof)
        if any(observed.get(edge) is False for edge in parts):
            continue
        for edge in parts:
            if edge not in observed:
                missing[edge] = None
                if len(missing) == width:
                    return list(missing)
    return list(missing)


def first_survivor(proofs, observed):
    return next((proof for proof in proofs if all(observed.get(edge) is not False for edge in edges(proof))), None)


def first_verified(proofs, observed):
    return next((proof for proof in proofs if all(observed.get(edge) is True for edge in edges(proof))), None)


def dispatch(arm, proofs, reader):
    observed = {}
    if arm in ("all_edges", "static9"):
        cap = WIDTH if arm == "static9" else None
        unseen = frontier(proofs, observed, cap)
        if unseen:
            observed.update(reader(unseen))
        return first_verified(proofs, observed), observed

    while (proof := first_survivor(proofs, observed)) is not None:
        missing = [edge for edge in edges(proof) if edge not in observed]
        if not missing:
            return proof, observed
        if arm == "adaptive9" and len(missing) + len(observed) > WIDTH:
            return first_verified(proofs, observed), observed
        if arm == "prefetch9":
            missing = frontier(proofs, observed, WIDTH)
        observed.update(reader(missing))
    return None, observed


class TopicReader:
    def __init__(self, reader, index, atoms, cutoff):
        self.reader = reader
        self.index = index
        self.atoms = atoms
        self.cutoff = cutoff
        self.values = {}
        self.clipped = 0
        self.forwards = 0

    def __call__(self, candidates):
        if set(candidates).intersection(self.values):
            raise ValueError("Duplicate topic verification within a request")
        for start in range(0, len(candidates), BATCH):
            chunk = candidates[start:start + BATCH]
            scores, clipped = self.reader.score_pairs(
                [(self.atoms[side][0], self.index.texts[doc]) for side, doc in chunk], task=live.TOPIC_TASK,
            )
            if len(scores) != len(chunk) or not np.isfinite(scores).all():
                raise ValueError("Incomplete topic judgments")
            self.values.update((edge, score > self.cutoff) for edge, score in zip(chunk, scores, strict=True))
            self.clipped += clipped
            self.forwards += 1
        return {edge: self.values[edge] for edge in candidates}


def evaluate(index, reader, panels, reference, depth, offset, cutoff):
    roles = RolePostings(index.catalogue)
    result = {}
    for publisher, subjects, queries in panels:
        result[publisher] = {}
        for name, conflicts in SHAPES.items():
            scores = {arm: Counter() for arm in ARMS}
            times = defaultdict(list)
            role_known = role_issued = 0
            edge_audit = Counter()
            for i, atoms in enumerate([queries[0], *queries]):
                start = perf_counter()
                ranks = index.rank([topic for topic, _ in atoms], depth=depth)
                plans = feasible(roles.personal, ranks, atoms, conflicts)
                common_ms = (perf_counter() - start) * 1000

                def known(proof):
                    return bool(proof) and all(topic in subjects.get(doc, ())
                                               for (topic, _), doc in zip(atoms, proof[1], strict=True))

                shift = (i + offset) % len(ARMS)
                per_query = {}
                for arm in ARMS[shift:] + ARMS[:shift]:
                    start = perf_counter()
                    checker = TopicReader(reader, index, atoms, cutoff)
                    proof, observed = dispatch(arm, plans, checker)
                    if proof:
                        # Confirm that every citation is bound both to its selected
                        # topic slot and to the same original declared contributor.
                        scoped = tuple({doc: rank[doc]} for doc, rank in zip(proof[1], ranks, strict=True))
                        if verified(index, list(dict.fromkeys(proof[1])), scoped, atoms, conflicts) != proof:
                            raise ValueError("Unconfirmed person or source in selected proof")
                    json.dumps({"matches": [proof] if proof else []}, separators=(",", ":"))
                    ms = common_ms + (perf_counter() - start) * 1000
                    per_query[arm] = (proof, observed, checker.forwards, checker.clipped, ms)
                if not i:
                    continue
                role_issued += bool(plans)
                role_known += known(plans[0] if plans else None)
                target, values, *_ = per_query["all_edges"]
                for (side, doc), positive in values.items():
                    annotated = atoms[side][0] in subjects.get(doc, ())
                    edge_audit.update(
                        annotated=int(annotated), annotated_pass=int(annotated and positive),
                        unjudged=int(not annotated), unjudged_pass=int(not annotated and positive),
                    )
                for arm, (proof, observed, forwards, clipped, ms) in per_query.items():
                    def replay(candidates):
                        return {edge: values[edge] for edge in candidates}

                    canonical, oracle_probes = dispatch(arm, plans, replay) if arm != "all_edges" else (target, values)
                    if arm in ("certificate", "prefetch9") and canonical != target:
                        raise AssertionError("The certified scheduler changed the full-reader output")
                    scores[arm].update(
                        known=known(proof), issued=bool(proof), pairs=len(observed), calls=forwards, clipped=clipped,
                        differs_from_all_edges=int(proof != target),
                        score_sign_changes=sum(value != values[edge] for edge, value in observed.items()),
                        canonical_pairs=len(oracle_probes), canonical_known=known(canonical),
                        canonical_differs_from_all_edges=int(canonical != target),
                    )
                    times[arm].append(ms)
            expected = reference[publisher][name]["metrics"]["role_exhaustive"]
            if (role_known, role_issued) != (expected["known"], expected["issued"]):
                raise ValueError(f"Previous nomination did not reproduce for {publisher}/{name}")
            result[publisher][name] = dict(
                queries=len(queries), role_known=role_known, role_issued=role_issued, edge_audit=dict(edge_audit),
                arms={arm: dict(**scores[arm], warm_p50_p95_ms=np.quantile(times[arm], [.50, .95]).tolist())
                      for arm in ARMS},
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reranker",
                 "catalogue_reference", "nomination_reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--cutoff", type=float, choices=(0., -4.), default=0.)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    expected = json.loads(args.catalogue_reference.read_bytes())["runs"][0]
    nominated = json.loads(args.nomination_reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    base = live.BASE
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
        or raw_identifiers(args.plos) & raw_identifiers(args.elife)):
        raise ValueError("Unexpected frozen inputs")
    plos_subjects = base.publisher_topics([args.plos])
    elife_subjects, elife_witness, _ = live.elife.build_requests(elife)["topics"]
    panels = (
        ("plos", plos_subjects, witness_queries(base.annotation_postings(plos, plos_subjects))),
        ("elife", elife_subjects, witness_queries(elife_witness)),
    )
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    topic_reader = Reranker(args.reranker)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        index = live.LocalIndex(merge(plos, elife), encoder, Path(scratch) / "mixed")
        results = evaluate(index, topic_reader, panels, nominated["panels"], nominated["depth"], args.order_offset, args.cutoff)
    write_json(args.output, dict(
        panels=results, order_offset=args.order_offset, cutoff=args.cutoff, gpu=torch.cuda.get_device_name(),
        embedding_checkpoint=checkpoint, reranker_revision=topic_reader.provenance["revision"],
        nomination_reference_sha256=digest(args.nomination_reference), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
