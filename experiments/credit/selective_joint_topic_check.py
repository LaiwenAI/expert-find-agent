"""Calibrate and restrict topic checking to viable cross-publication proofs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


BASE = live.BASE
MODES = ("rank", "full", "viable", "top_pair")
FOLDS = 3
UNJUDGED_LIMIT = .20
MIN_CALIBRATION_ISSUED = 5


def assess(index, reader, payload, mode):
    start = perf_counter()
    atoms = tuple((obj["topic"], obj["role"]) for obj in json.loads(payload)["requirements"])
    ranks = index.rank([a[0] for a in atoms])
    plan, loaded = live.inspect(index, atoms, ranks, "joint")
    proofs = BASE.choose_proofs(BASE.read_articles(SimpleNamespace(ownership=loaded), plan, atoms, ranks), ranks)
    if mode == "top_pair":
        proofs = proofs[:1]
    keys = sorted({(i, option[i+1]) for option in proofs for i in range(2)})
    if mode == "full":
        keys = [(i, doi) for i in range(2) for doi in plan]
    values = {}
    checks = 0
    if mode == "rank":
        values = {key: -float(ranks[key[0]][key[1]]) for key in keys}
    elif keys:
        scores, clipped = reader.score_pairs(
            [(atoms[i][0], index.texts[doi]) for i, doi in keys], task=live.TOPIC_TASK,
        )
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Invalid topic reader result")
        checks = len(keys)
        values = dict(zip(keys, scores, strict=True))

    def margins(option):
        return tuple(values[i, doi] for i, doi in enumerate(option[1:]))

    # Preserve rank ordering when verifier margins tie.
    candidate = max(proofs, key=lambda p: (min(margins(p)), sum(margins(p))), default=None)
    confidence = None if candidate is None else min(margins(candidate))
    result = json.dumps({"candidate": candidate, "confidence": confidence})
    return result, (perf_counter() - start) * 1000, checks


def collect(index, reader, subjects, cases, offset, *, audit_rows=None):
    for mode in MODES:
        assess(index, reader, live.request(cases[0]), mode)
    output = []
    for number, atoms in enumerate(cases):
        key = "|".join(topic for topic, _ in atoms)
        payload = live.request(atoms)
        position = (number + offset) % len(MODES)
        results = {}
        candidates = {} if audit_rows is not None else None
        for mode in MODES[position:] + MODES[:position]:
            encoded, latency, checks = assess(index, reader, payload, mode)
            item = json.loads(encoded)
            match = item["candidate"]
            if candidates is not None:
                candidates[mode] = match
            reference = bool(match) and all(atoms[i][0] in subjects[match[i+1]] for i in range(2))
            results[mode] = dict(known=reference, checks=checks, latency_ms=latency,
                                 confidence=item["confidence"])
        output.append((key, results))
        if audit_rows is not None:
            audit_rows.append((atoms, candidates))
    return output


def cutoff(rows, mode):
    """Freeze the most permissive useful cutoff under the calibration proxy."""
    values = [results[mode] for _, results in rows if results[mode]["confidence"] is not None]
    candidates = sorted({item["confidence"] for item in values})
    for threshold in candidates:
        admitted = [r for r in values if r["confidence"] >= threshold]
        count = len(admitted)
        if count >= MIN_CALIBRATION_ISSUED and sum(not r["known"] for r in admitted) <= UNJUDGED_LIMIT * count:
            return threshold
    return None


def qualifies(record, cutoff):
    confidence = record["confidence"]
    return confidence is not None and cutoff is not None and confidence >= cutoff


def score_panel(rows, mode, cuts):
    stats = Counter()
    for (_, estimates), threshold in zip(rows, cuts, strict=True):
        record = estimates[mode]
        admitted = qualifies(record, threshold)
        stats.update(known=int(admitted and record["known"]), issued=int(admitted),
                     tag_unresolved=int(admitted and not record["known"]))
    return dict(stats)


def matched_issuance(rows, count, mode):
    # Diagnostic: match the full gate's observed issuance, without using labels.
    if count <= 0:
        return [None] * len(rows)
    ranking = sorted(((-row[mode]["confidence"], j) for j, (_, row) in enumerate(rows)
                      if row[mode]["confidence"] is not None))[:count]
    chosen = {j for _, j in ranking}
    return [-float("inf") if j in chosen else None for j in range(len(rows))]


def report(plos, panels):
    def fold(group):
        return int(sha256(group.encode()).hexdigest()[:8], 16) % FOLDS

    thresholds = {
        mode: [cutoff([r for r in plos if fold(r[0]) != i], mode) for i in range(FOLDS)]
        for mode in MODES
    }
    transfer = {mode: cutoff(plos, mode) for mode in MODES}
    out = {}
    for name, rows in panels.items():
        out[name] = dict(queries=len(rows), topic_pairs=len({key for key, _ in rows}), arms={})
        for mode in MODES:
            timing = [r[mode]["latency_ms"] for _, r in rows]
            locked = [thresholds[mode][fold(group)] if name == "plos" else transfer[mode]
                      for group, _ in rows]
            selected = score_panel(rows, mode, locked)
            out[name]["arms"][mode] = dict(
                calibrated=selected,
                unfiltered=score_panel(rows, mode, [-float("inf")] * len(rows)),
                zero_cutoff=score_panel(rows, mode, [1e-12] * len(rows)) if mode != "rank" else None,
                checks=sum(r[mode]["checks"] for _, r in rows),
                model_requests=sum(r[mode]["checks"] > 0 for _, r in rows),
                p50_ms=float(np.quantile(timing, .50)),
                p95_ms=float(np.quantile(timing, .95)),
            )
        target = out[name]["arms"]["full"]["calibrated"]["issued"]
        if target:
            cuts = {mode: matched_issuance(rows, target, mode) for mode in MODES}
            matched_flags = {mode: np.array([
                int(qualifies(estimates[mode], cut) and estimates[mode]["known"])
                for (_, estimates), cut in zip(rows, cuts[mode], strict=True)
            ]) for mode in MODES}
            groups = [group for group, _ in rows]
            for mode in MODES:
                out[name]["arms"][mode]["matched_to_full_issuance"] = score_panel(rows, mode, cuts[mode])
                out[name]["arms"][mode]["matched_delta_vs_full"] = clustered_paired_interval(
                    matched_flags[mode] - matched_flags["full"], groups,
                )
    return dict(calibration_proxy_limit=UNJUDGED_LIMIT, fold_cutoffs=thresholds,
                transfer_cutoffs=transfer, panels=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reranker", "output"):
        p.add_argument(arg, type=Path)
    p.add_argument("--order-offset", type=int, default=0)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU compute node")
    checkpoint = live.verify_embedding(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    plos, _ = BASE.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    plos_subjects = BASE.publisher_topics([args.plos_xml])
    life_subjects, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, BASE.query_panel(BASE.annotation_postings(plos, plos_subjects))),
                  elife=(life, life_subjects, life_cases))
    collected = {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, subjects, queries) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            collected[name] = collect(index, reader, subjects, queries, args.order_offset)
            print(f"Completed {name}: {len(queries)}", flush=True)
    write_json(args.output, dict(
        **report(collected["plos"], collected), order_offset=args.order_offset, embedding_checkpoint=checkpoint,
        reranker_checkpoint=reader.provenance["revision"], elife_projection_sha256=checksum,
        script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    main()
