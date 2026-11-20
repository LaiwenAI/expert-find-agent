"""Complete verified joint-prefix proofs under a three-source budget."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_role_postings as postings
from experiments.credit import live_joint_source_request as live
from experiments.credit import robust_role_postings as stress
from experiments.followir.core17_reranking import digest, write_json


CONFIGURATIONS = ((0, 0), (50, 0), (50, 20), (100, 0))
CONTROLS = ("staged_byline", "staged_roles", "certified_roles")


def evidence(loaded, atoms, ranks):
    return live.BASE.read_articles(SimpleNamespace(ownership=loaded), loaded.keys(), atoms, ranks)


def proof_key(proof, ranks):
    a, b = ranks[0][proof[1]], ranks[1][proof[2]]
    return max(a, b), a + b, proof


def alternative_proofs(observed, ranks, byline, anchors):
    """List proofs that one more declaration could supply, assuming any role."""
    viable = []
    pool = set().union(*ranks) - set(anchors)
    for person, sides in observed.items():
        for missing in range(2):
            if sides[1 - missing]:
                anchor = min(sides[1 - missing], key=ranks[1 - missing].__getitem__)
                for doc in ranks[missing]:
                    if doc in pool and person in byline[doc]:
                        proof = (person, doc, anchor) if missing == 0 else (person, anchor, doc)
                        viable.append((proof_key(proof, ranks), doc, missing, person))
    return viable


def record(proof, atoms, ranks, subjects):
    if proof is None:
        return dict(match=None, priority=None, known=False)
    return dict(
        match=list(proof), priority=proof_key(proof, ranks)[:2],
        known=all(topic in subjects[doi] for (topic, _), doi in zip(atoms, proof[1:], strict=True)),
    )


def nominate(index, ranks):
    initial, extra = live.simple_order(ranks)
    return live.joint.posting_plan(
        index.catalogue.byline, set().union(*ranks), ranks, [*initial, extra[0]], live.DEPTH,
    )


def fixed(index, atoms, ranks):
    plan = nominate(index, ranks)
    loaded = index.open_sources(plan)
    proofs = live.BASE.choose_proofs(evidence(loaded, atoms, ranks), ranks)
    return plan, next(iter(proofs), None)


def gap_source(viable, projected, atoms, default):
    if not viable:
        return default

    def priority(option):
        key, doi, side, person = option
        if projected is None:
            return 0, key
        fields = projected.get(doi, {}).get(person, set())
        expected = atoms[side][1]
        score = 0 if expected in fields else (1 if not fields else 2)
        return score, key

    option = min(viable, key=priority)
    return option[1]


def staged(index, atoms, ranks, plan, projected, *, certify):
    anchors = plan[:2]
    loaded = index.open_sources(anchors)
    observed = evidence(loaded, atoms, ranks)
    prefix = next(iter(live.BASE.choose_proofs(observed, ranks)), None)
    viable = alternative_proofs(observed, ranks, index.catalogue.byline, anchors) if certify or prefix is None else []
    if certify and (not viable or (
        prefix is not None and proof_key(prefix, ranks) <= min(option[0] for option in viable)
    )):
        return prefix, 2, plan[2], prefix is not None

    third = plan[2] if prefix is not None else gap_source(viable, projected, atoms, plan[2])
    loaded.update(index.open_sources([third]))
    if len(loaded) != 3:
        raise ValueError("Unexpected repeated declaration")
    options = live.BASE.choose_proofs(evidence(loaded, atoms, ranks), ranks)
    return next(iter(options), None), 3, third, prefix is not None


def evaluate(index, subjects, queries, offset):
    source_index = postings.RolePostings(index.catalogue)
    bank = []
    fixed_time = []
    for atoms in queries:
        ranks = index.rank([topic for topic, _ in atoms])
        start = perf_counter_ns()
        plan, proof = fixed(index, atoms, ranks)
        fixed_time.append((perf_counter_ns() - start) / 1e6)
        bank.append((atoms, ranks, plan, record(proof, atoms, ranks, subjects)))
    scenarios = {}
    for coverage, corruption in CONFIGURATIONS:
        masks = stress.MASKS if (coverage, corruption) not in ((0, 0), (100, 0)) else ("",)
        variants = []
        for mask in masks:
            projected, inventory = stress.project(source_index, index.catalogue, coverage, corruption, mask)
            rows = {method: [] for method in CONTROLS}
            counts = {method: Counter() for method in CONTROLS}
            times = defaultdict(list)
            for j, (atoms, ranks, plan, control) in enumerate(bank):
                shift = (j + offset) % len(CONTROLS)
                replies = {}
                for method in CONTROLS[shift:] + CONTROLS[:shift]:
                    start = perf_counter_ns()
                    answer, reads, third, prefix_proof = staged(
                        index, atoms, ranks, nominate(index, ranks), projected if method != "staged_byline" else None,
                        certify=method == "certified_roles",
                    )
                    times[method].append((perf_counter_ns() - start) / 1e6)
                    replies[method] = answer
                    result = record(answer, atoms, ranks, subjects)
                    counts[method].update(
                        reads=reads, two_read_stops=int(reads == 2),
                        stops_with_proof=int(reads == 2 and prefix_proof),
                        prefix_proofs=int(prefix_proof), changed_third=int(reads == 3 and third != plan[2]),
                        lost_known=int(control["known"] and not result["known"]),
                        gained_known=int(result["known"] and not control["known"]),
                    )
                    rows[method].append(dict(
                        atoms=atoms, topic_pair=json.dumps([topic for topic, _ in atoms]),
                        arms=dict(owner_gap=control, joint=result),
                    ))
                if replies["certified_roles"] != replies["staged_roles"]:
                    raise AssertionError("A stopping certificate changed the reference response")
            variants.append(dict(
                mask=mask, inventory=inventory, arms=counts, comparisons=stress.compare(rows),
                p95_selection_and_reads_ms={arm:float(np.quantile(samples, .95)) for arm,samples in times.items()},
            ))
        scenarios[f"{coverage}-{corruption}"] = variants
    return dict(
        queries=len(bank), fixed_known=sum(item[-1]["known"] for item in bank),
        fixed_issued=sum(item[-1]["match"] is not None for item in bank),
        fixed_p95_selection_and_reads_ms=float(np.quantile(fixed_time, .95)), scenarios=scenarios,
    )



def live_latency(index, cases, offset):
    """Time paired local requests from JSON parsing through response encoding."""
    methods = ("fixed_triple", "certified_prefix")
    timings = {method: [] for method in methods}
    successful = 0

    def handle(payload, method):
        start = perf_counter_ns()
        atoms = tuple((item["topic"], item["role"]) for item in json.loads(payload)["requirements"])
        ranks = index.rank([topic for topic, _ in atoms])
        if method == "fixed_triple":
            _, proof = fixed(index, atoms, ranks)
        else:
            proof, _, _, _ = staged(index, atoms, ranks, nominate(index, ranks), None, certify=True)
        response = json.dumps({"matches": [proof] if proof else []}, separators=(",", ":"))
        return response, (perf_counter_ns() - start) / 1e6

    for method in methods:
        handle(live.request(cases[0]), method)
    for i, atoms in enumerate(cases):
        payload = live.request(atoms)
        outputs = {}
        order = methods if (i + offset) % 2 == 0 else methods[::-1]
        for method in order:
            response, duration = handle(payload, method)
            outputs[method] = response
            timings[method].append(duration)
        successful += outputs[methods[0]] == outputs[methods[1]]
    if successful != len(cases):
        raise AssertionError("The live prefix route changed a fixed-triple response")
    return dict(equal_responses=successful, p95_ms={
        method: float(np.quantile(samples, .95)) for method, samples in timings.items()
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"] or live.DEPTH != expected["query_depth"]
    ):
        raise ValueError("Frozen inputs differ from reference")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_cases = base.query_panel(base.annotation_postings(plos, plos_subjects))
    life_subjects, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, plos_cases), elife=(life, life_subjects, life_cases))
    encoder = QwenTorchEmbedding(args.embedding)
    results = {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, subjects, queries) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            results[name] = evaluate(index, subjects, queries, args.order_offset)
            results[name]["live_latency"] = live_latency(index, queries, args.order_offset)
            reference = expected["panels"][name]["arms"]["structured"]["joint"]
            if (results[name]["fixed_known"], results[name]["fixed_issued"]) != (reference["known"], reference["issued"]):
                raise ValueError(f"Fixed-plan control did not reproduce on {name}")
    write_json(args.output, dict(
        order_offset=args.order_offset, gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)), panels=results,
    ))


if __name__ == "__main__":
    main()
