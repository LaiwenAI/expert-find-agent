"""Replay frozen paired-role decisions under prospective loopback load."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.budgeted_source_role_search import role_check
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.elife_role_prompt_service import arrivals_for, replay, summarize
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates, phrase_bank
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.joint_request_queue import Request
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.lightweight_paired_role import METHODS as LIGHTWEIGHT_METHODS, decide, fit_lightweight
from experiments.credit.paired_role_verification import PairedReader, alternative_roles, apply, proposals
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_factorial_diagnosis import payload, plan
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("baseline", "batch", "reserve_agreement")
PRESCREEN_ARMS = ("baseline", "reserve_agreement", "reserve_prescreen")
CASCADE_ARMS = ("baseline", "reserve_prescreen", "student_then_reranker", "student_then_teacher")
JOINT_SERVICE_ARMS = ("baseline", "reserve_prescreen", "student_then_teacher", "dense_joint")
RATES = (28, 60)
FROZEN_SHA = "f3f24de1b725c529b6dc142a0eedfc6413129bc67643e9b74d0909df0b94c956"


def workload(index, cases, wording):
    table = phrase_bank(wording)
    labels = {}
    for case in cases:
        texts = [index.articles[paper]["clue"] for paper in case["papers"]]
        tasks = [table[role][int(case["family"][:16], 16) % len(table[role])].strip().rstrip(".?!")
                 for role in case["roles"]]
        for view, clues in (("keywords", list(map(index.keywords, texts))), ("excerpt", texts)):
            if not all(clues):
                raise ValueError("Empty topic clue")
            req = Request("elife", json.dumps({"requirements": [
                dict(topic=a, role=b) for a, b in zip(clues, tasks, strict=True)]}))
            if req in labels:
                raise ValueError("Expected distinct replay requests")
            labels[req] = case | {"view": view}
    return labels


def execute(service, request, arm, cutoff, lightweight=None):
    if arm == "baseline":
        return service.search(request, "screened_one_call")
    requirements = json.loads(request.payload)["requirements"]
    clues = [item["topic"] for item in requirements]
    roles = [item["role"] for item in requirements]
    vectors = service.encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                     [f"Instruct: {TASK}\nQuery:{role}" for role in roles])
    scores = {name: vectors[2:] @ service.banks[name].T for name in service.bands}
    ranking = service.index.full_rank(vectors[:2], clues, None)
    state = plan(service, ranking, scores, clues, roles, None, "screened", run_role=False)
    pair, base, response = state["pair"], state["roles"], state["response"]
    if arm == "dense_joint":
        if not pair and not response["calls"]:
            from experiments.credit.paired_role_joint_postings import dense_source_fallback

            return json.dumps(dense_source_fallback(service, ranking, scores, clues))
        arm = "student_then_teacher"
    if not pair or response["calls"] or response["owners"]:
        return json.dumps(response)
    if arm in ("reserve_agreement", "reserve_prescreen", *LIGHTWEIGHT_METHODS, *CASCADE_ARMS[2:]):
        checked, forwards, tokens, _ = role_check(service.index, pair, scores, service.bands,
                                                  service.role, roles, service.cutoffs["role"])
        if forwards or candidates(service.index, pair, checked):
            return json.dumps(payload(service.index, pair, checked, role_calls=forwards, tokens=tokens))
    if arm in CASCADE_ARMS[2:]:
        heads, definitions, fit = lightweight
        student = decide("reserve_student", service, vectors[2:], roles, scores, pair, base, heads,
                         definitions, fit["reserve_student"]["cutoff"])
        if student["owners"]:
            return json.dumps(student)
        arm = "reserve_reranker" if arm == "student_then_reranker" else "reserve_prescreen"
    if arm in LIGHTWEIGHT_METHODS:
        heads, definitions, fit = lightweight
        return json.dumps(decide(arm, service, vectors[2:], roles, scores, pair, base, heads,
                                 definitions, fit[arm]["cutoff"]))
    if arm == "reserve_prescreen":
        pools = alternative_roles(scores)
        eligible = [pool[0] for pool in pools]
        # Dense agreement admits only this label pair. With no owner for
        # that pair, the role reader cannot change what gets issued.
        if cutoff is None or not candidates(service.index, pair, [{role} for role in eligible]):
            return json.dumps(response)
        choice, margin, tokens = service.role.batched(roles, pools)
        observation = dict(choice=choice if choice == eligible else None, margin=margin, forwards=1, tokens=tokens)
        return json.dumps(apply(service.index, pair, base, observation, cutoff))
    methods = ("two_batched", "top_one") if arm == "reserve_agreement" else ("two_batched",)
    observations = proposals(service.index, pair, base, roles, scores, service.role, methods)
    batch = observations["two_batched"]
    if arm == "reserve_agreement" and batch["choice"] != observations["top_one"]["choice"]:
        batch = batch | {"choice": None}
    return json.dumps(apply(service.index, pair, base, batch, cutoff))


def clusters(cases, articles):
    groups = []
    for row in cases:
        docs = tuple(row["papers"])
        owners = set().union(*(articles[doc]["roles"] for doc in docs))
        identities = owners | set(docs)
        existing = [n for n, (_, IDs) in enumerate(groups) if identities & IDs]
        families = {row["family"]}
        for n in reversed(existing):
            merged_families, merged_ids = groups.pop(n)
            identities.update(merged_ids)
            families.update(merged_families)
        groups.append((families, identities))
    return {f: min(families) for families, _ in groups for f in families}


def summarize_proofs(responses, labels):
    metrics = {}
    for req, proof in responses.items():
        req_label = labels[req]
        papers, owners, roles = proof
        record = evidence(dict(papers=papers, owners=owners, roles=roles, calls=0, source_calls=0,
                               role_calls=0), req_label)
        metrics[req] = {key: record[key] for key in ("strict", "issued", "adverse", "wrong_source")}
    return metrics


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    if digest(args.paired_frozen) != FROZEN_SHA:
        raise ValueError("Paired calibration record changed")
    frozen = json.loads(args.paired_frozen.read_bytes())["runs"][0]
    cutoff = frozen["fit"]["two_batched"]["cutoff"]
    configuration = ("niso", "training", "bands", "source_frozen", "role_frozen")
    for name in configuration:
        if digest(getattr(args, name)) != frozen["inputs"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    manifest = json.loads(args.manifest.read_bytes())
    articles, cases, census = build(args.projection, manifest)
    selected = sample_families(cases)
    if sorted({row["family"] for row in selected}) != manifest["chosen_families"]:
        raise ValueError("Changed selection of test families")
    definitions, training, wording = queries(args.niso, args.training)
    embedding = verify_checkpoint(args.embedding)["revision"]
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    reader = PairedReader(args.role_reader, definitions)
    reader.definitions = ROLES
    revisions = dict(embedding_revision=embedding, source_reader_revision=source.provenance["revision"],
                     role_reader_revision=reader.provenance["revision"])
    if any(value != frozen[name] for name, value in revisions.items()):
        raise ValueError("Checkpoint differs from the frozen comparison")
    index = PostingCatalog(articles, encoder)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    service = Service(index, encoder, banks, bands, source, reader, frozen["cutoffs"])
    labels = workload(index, selected, wording)
    if args.stage_audit_reference is not None:
        from experiments.credit.paired_role_stage_audit import audit

        audit(service, labels, cutoff, args.stage_audit_reference, args.output, args.reverse)
        return
    if args.source_postings_reference is not None:
        from experiments.credit.paired_role_source_postings import run_postings

        run_postings(service, labels, cutoff, args.source_postings_reference, args.output, args.reverse)
        return
    if args.joint_postings_reference is not None:
        from experiments.credit.paired_role_joint_postings import run_joint

        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, _, fit = fit_lightweight(args.plos, service, definitions, training)
        run_joint(service, labels, cutoff, heads, fit["reserve_student"]["cutoff"],
                  args.joint_postings_reference, args.output, args.reverse)
        return
    by_arm, results = {}, {}
    lightweight = report = None
    requested = PRESCREEN_ARMS if args.prescreen else ARMS
    if args.plos:
        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, training_report, fit = fit_lightweight(args.plos, service, definitions, training)
        report = dict(training=training_report, calibration=fit, plos_sha256=digest(args.plos))
        lightweight = (heads, definitions, fit)
        if args.joint_postings_service:
            requested = JOINT_SERVICE_ARMS
        else:
            requested = CASCADE_ARMS if args.cascade else ("baseline", "reserve_prescreen", *LIGHTWEIGHT_METHODS)
        print("Frozen lightweight routes", fit, flush=True)
    arms = requested[::-1] if args.reverse else requested
    for offset, rate in enumerate(RATES):
        trace, arrivals = arrivals_for(labels, rate)
        results[rate] = {}
        for arm in arms[offset:] + arms[:offset]:
            records, elapsed = replay(lambda request: execute(service, request, arm, cutoff, lightweight),
                                      trace, arrivals, trace[:24])
            detail, responses = summarize(trace, arrivals, records, elapsed, labels, 2)
            if detail["failures"] or len(responses) != len(labels):
                raise ValueError(f"Incomplete replay for {arm} at {rate}: {detail['failures']}")
            calls = Counter(role_forwards=0, source_forwards=0)
            for record in records:
                response = json.loads(record["response"])
                if response["role_calls"] + response["source_calls"] != response["calls"] or response["calls"] > 1:
                    raise ValueError("Shared verifier budget exceeded")
                calls.update(role_forwards=response["role_calls"], source_forwards=response["source_calls"])
                if args.joint_postings_service:
                    calls.update(source_rows=response.get("source_rows", response["source_calls"]))
            detail["verification"] = dict(calls)
            if arm in by_arm and by_arm[arm] != responses:
                raise ValueError("Traffic changed the emitted evidence")
            by_arm[arm] = responses
            results[rate][arm] = detail
            print(rate, arm, detail["counts"], round(detail["p95_ms"]["total_ms"], 2), flush=True)
    if args.prescreen and not args.plos and by_arm["reserve_agreement"] != by_arm["reserve_prescreen"]:
        changed = sum(by_arm["reserve_agreement"][req] != by_arm["reserve_prescreen"][req] for req in labels)
        raise ValueError(f"Agreement prescreen changed {changed} evidence payloads")
    groups = clusters(selected, articles)
    metrics = {arm: summarize_proofs(proofs, labels) for arm, proofs in by_arm.items()}
    contrasts = {}
    for arm in requested:
        contrasts[arm] = {}
        for view in ("keywords", "excerpt"):
            rows = [req for req, label in labels.items() if label["view"] == view]
            positives = [req for req in rows if labels[req]["gold"]]
            group_ids = [groups[labels[req]["family"]] for req in positives]
            counts = Counter()
            for req in rows:
                counts.update(metrics[arm][req])
            delta = [int(metrics[arm][req]["strict"]) - int(metrics["baseline"][req]["strict"]) for req in positives]
            interval = clustered_paired_interval(delta, group_ids)
            interval["sampling_unit"] = "connected paper/recorded-author groups"
            contrasts[arm][view] = dict(totals=dict(counts), positives=len(positives),
                                        requests=len(rows), paired_strict_change=interval)
            if arm == "dense_joint":
                extra = [int(metrics[arm][req]["strict"]) - int(metrics["student_then_teacher"][req]["strict"])
                         for req in positives]
                paired = clustered_paired_interval(extra, group_ids)
                paired["sampling_unit"] = "connected paper/recorded-author groups"
                contrasts[arm][view]["paired_strict_change_vs_cascade"] = paired
    signatures = {
        arm: sha256(json.dumps([proofs[req] for req in labels], ensure_ascii=False).encode()).hexdigest()
        for arm, proofs in by_arm.items()
    }
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection),
        manifest_sha256=digest(args.manifest), paired_results_sha256=digest(args.paired_frozen),
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, prescreen=args.prescreen, census=census,
        distinct_requests=len(labels), author_connected_groups=len(set(groups.values())),
        cutoffs=frozen["cutoffs"] | {"two_batched": cutoff}, results=results, quality=contrasts,
        response_signatures=signatures, lightweight=report, cascade=args.cascade, **revisions,
        **({"joint_postings_service": True} if args.joint_postings_service else {}),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "paired_frozen", "niso", "training", "bands",
                 "source_frozen", "role_frozen", "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prescreen", action="store_true")
    parser.add_argument("--plos", type=Path)
    parser.add_argument("--cascade", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--stage-audit-reference", type=Path)
    diagnostics.add_argument("--source-postings-reference", type=Path)
    diagnostics.add_argument("--joint-postings-reference", type=Path)
    diagnostics.add_argument("--joint-postings-service", action="store_true")
    args = parser.parse_args()
    if args.cascade and not args.plos:
        parser.error("--cascade requires --plos")
    if args.joint_postings_reference is not None and args.plos is None:
        parser.error("--joint-postings-reference requires --plos")
    if args.joint_postings_service and args.plos is None:
        parser.error("--joint-postings-service requires --plos")
    run(args)
