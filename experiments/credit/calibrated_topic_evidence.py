"""Calibrate article-topic evidence on official judgments before transfer."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import live_joint_source_request as live
from experiments.credit.live_joint_role_postings import RolePostings
from experiments.credit.mixed_publisher_joint import merge, raw_identifiers
from experiments.credit.multi_qualification_proofs import SHAPES, witness_queries
from experiments.credit.topic_proof_certification import feasible, first_verified, frontier
from experiments.credit.trec_topic_panel import normalize_title
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


TASKS = {
    "original": live.TOPIC_TASK,
    "scope": (
        "Decide whether the article substantively studies the named research subject. "
        "Use the title and abstract, accept equivalent scientific wording, and reject only incidental mentions."
    ),
}
BATCH = 12


def margins(reader, pairs, task):
    result = []
    clipped = 0
    for i in range(0, len(pairs), BATCH):
        rows = pairs[i:i + BATCH]
        values, count = reader.score_pairs(rows, task=task)
        if len(values) != len(rows) or not np.isfinite(values).all():
            raise ValueError("Incomplete topic judgments")
        result.extend(values)
        clipped += count
    return np.array(result), clipped


def operating_point(values, labels):
    negative = np.sort(values[labels == 0])[::-1]
    return float(negative[len(negative) // 10])


def grade_report(values, rows, cutoff):
    result = {}
    for grade in (0, 1, 2):
        chosen = [(int(score > cutoff), row["topic"])
                  for score, row in zip(values, rows, strict=True) if row["grade"] == grade]
        indicators, groups = zip(*chosen, strict=True)
        result[str(grade)] = dict(passed=sum(indicators), rows=len(indicators),
                                  pass_fraction=clustered_paired_interval(indicators, groups))
    return result


def calibrate(panel, encoder, reader):
    reports = {}
    cuts = {}
    for split, records in panel["splits"].items():
        topics = sorted({row["topic"] for row in records})
        qids = {topic: i for i, topic in enumerate(topics)}
        queries = [panel["topics"][str(topic)] for topic in topics]
        texts = [row["title"] + "\n" + row["abstract"] for row in records]
        vectors = encoder.encode(queries, query=True)
        docs = encoder.encode(texts)
        similarities = np.einsum("ij,ij->i", vectors[[qids[row["topic"]] for row in records]], docs)
        pairs = [(panel["topics"][str(row["topic"])], text) for row, text in zip(records, texts, strict=True)]
        observed = {name: margins(reader, pairs, task) for name, task in TASKS.items()}
        observed["cosine"] = (similarities, 0)
        reports[split] = {}
        for name, (scores, clipped) in observed.items():
            labels = np.asarray([row["grade"] for row in records])
            if split == "calibration":
                cuts[name] = operating_point(scores, labels)
            reports[split][name] = dict(cutoff=cuts[name], clipped=clipped,
                                        by_grade=grade_report(scores, records, cuts[name]))
        reports[split]["original_zero"] = grade_report(observed["original"][0], records, 0.)
    return reports, cuts


def project_outputs(index, reader, subjects, queries, cuts, depth):
    postings = RolePostings(index.catalogue).personal
    results = {shape: defaultdict(Counter) for shape in SHAPES}
    for atoms in queries:
        topics = [topic for topic, _ in atoms]
        ranks = index.rank(topics, depth=depth)
        vectors = index.encoder.encode(topics, query=True)
        docs = {doc: i for i, doc in enumerate(index.ids)}
        for shape, conflicts in SHAPES.items():
            plans = feasible(postings, ranks, atoms, conflicts)
            checks = frontier(plans, {}, None)
            prompts = [(topics[side], index.texts[doc]) for side, doc in checks]
            shared = {name: margins(reader, prompts, task) for name, task in TASKS.items()}
            shared["cosine"] = (np.asarray([vectors[side] @ index.embeddings[docs[doc]]
                                          for side, doc in checks]), 0)
            shared["original_zero"] = shared["original"]
            shared["original_minus4"] = shared["original"]
            for name, (scores, clipped) in shared.items():
                cutoff = {"original_zero": 0., "original_minus4": -4.}.get(name, cuts.get(name))
                accepted = dict(zip(checks, (bool(value > cutoff) for value in scores), strict=True))
                proof = first_verified(plans, accepted)
                count = results[shape][name]
                count.update(pairs=len(checks), clipped=clipped, issued=bool(proof))
                count["known"] += bool(proof) and all(topic in subjects.get(doc, ())
                                                    for topic, doc in zip(topics, proof[1], strict=True))
                for (side, doc), positive in accepted.items():
                    annotated = topics[side] in subjects.get(doc, ())
                    count.update(annotated=int(annotated), annotated_pass=int(annotated and positive),
                                 unjudged=int(not annotated), unjudged_pass=int(not annotated and positive))
    return {shape: dict(measures) for shape, measures in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "panel_manifest", "plos", "fixture", "elife", "elife_manifest",
                  "embedding", "reranker", "catalogue_reference", "proof_reference", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    expected = json.loads(args.catalogue_reference.read_bytes())["runs"][0]
    proof = json.loads(args.proof_reference.read_bytes())["runs"][0]
    panel = json.loads(args.panel.read_bytes())
    if digest(args.panel) != json.loads(args.panel_manifest.read_bytes())["panel_sha256"]:
        raise ValueError("Changed TREC reference panel")
    checkpoint = live.verify_embedding(args.embedding)
    base = live.BASE
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
        or raw_identifiers(args.plos) & raw_identifiers(args.elife)):
        raise ValueError("Unexpected frozen publisher inputs")
    published = {normalize_title(title) for catalog in (plos, elife) for title, _ in catalog.documents.values()}
    for rows in panel["splits"].values():
        if any(normalize_title(row["title"]) in published for row in rows):
            raise ValueError("TREC and publisher document titles intersect")
    subjects = base.publisher_topics([args.plos])
    elife_subjects, elife_witness, _ = live.elife.build_requests(elife)["topics"]
    work = (
        ("plos", subjects, witness_queries(base.annotation_postings(plos, subjects))),
        ("elife", elife_subjects, witness_queries(elife_witness)),
    )
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    calibration, cuts = calibrate(panel, encoder, reader)
    transfer = {}
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        index = live.LocalIndex(merge(plos, elife), encoder, Path(temporary) / "mixed")
        for publisher, subjects, queries in work:
            transfer[publisher] = project_outputs(index, reader, subjects, queries, cuts, 128)
            for shape, arms in transfer[publisher].items():
                prior = proof["panels"][publisher][shape]["edge_audit"]
                if any(arms["original_zero"][key] != count for key, count in prior.items()):
                    raise ValueError("Previously observed default threshold did not reproduce")
    write_json(args.output, dict(
        calibration=calibration, cutoffs=cuts, transfer=transfer, gpu=torch.cuda.get_device_name(),
        embedding_checkpoint=checkpoint, reader_revision=reader.provenance["revision"],
        panel_sha256=digest(args.panel), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
