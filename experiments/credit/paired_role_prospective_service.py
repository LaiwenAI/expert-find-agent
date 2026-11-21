"""Replay frozen paired-role decisions under prospective loopback load."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.distilled_role_recheck import route
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.elife_role_prompt_service import arrivals_for, replay, summarize
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates, phrase_bank
from experiments.credit.engineering_one_role_recheck import replace
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
MATCHED_SERVICE_ARMS = ("baseline", "student_then_teacher", "dense_joint", "matched_resident")
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


def request_inputs(request):
    requirements = json.loads(request.payload)["requirements"]
    clues = [item["topic"] for item in requirements]
    roles = [item["role"] for item in requirements]
    prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
    prompts.extend(f"Instruct: {TASK}\nQuery:{role}" for role in roles)
    return clues, roles, prompts


@dataclass
class RoleCheck:
    phrases: list
    pools: list
    finish: Callable


def finish_role(service, result):
    if isinstance(result, RoleCheck):
        if len(result.phrases) == 1:
            role, gap, tokens = service.role.read(result.phrases[0], result.pools[0])
            return result.finish(([role], gap, tokens))
        return result.finish(service.role.batched(result.phrases, result.pools))
    return result


def execute(service, request, arm, cutoff, lightweight=None, resident=None, *, withheld=None, inspect_source=None, source_guard=None):
    if arm == "baseline":
        if withheld is not None:
            raise ValueError("The original single-slot arm does not support corpus masking")
        return service.search(request, "screened_one_call")
    clues, roles, prompts = request_inputs(request)
    result = resolve(service, arm, cutoff, lightweight, resident, clues, roles, service.encoder.encode(prompts),
                     withheld=withheld, inspect_source=inspect_source, source_guard=source_guard)
    return finish_role(service, result)


def execute_group(service, requests, arm, cutoff, lightweight, resident, stage):
    inputs = [request_inputs(request) for request in requests]
    if stage == "embedding":
        flat = service.encoder.encode([prompt for _, _, prompts in inputs for prompt in prompts], group_size=4)
        responses = [finish_role(service, resolve(service, arm, cutoff, lightweight, resident,
                                                 clues, roles, flat[4*i:4*i + 4]))
                     for i, (clues, roles, _) in enumerate(inputs)]
        yield from enumerate(responses)
        return

    deferred, completed = [], []
    for i, (clues, roles, prompts) in enumerate(inputs):
        result = resolve(service, arm, cutoff, lightweight, resident, clues, roles, service.encoder.encode(prompts))
        if isinstance(result, RoleCheck) and (stage == "mixed_role" or len(result.phrases) == 2):
            deferred.append((i, result))
            continue
        response = finish_role(service, result)
        if stage == "mixed_role":
            yield i, response
        else:
            completed.append((i, response))

    if len(deferred) == 1:
        i, task = deferred[0]
        completed.append((i, finish_role(service, task)))
    elif deferred:
        verified = service.role.batched_requests([(task.phrases, task.pools) for _, task in deferred])
        for j, ((i, task), outcome) in enumerate(zip(deferred, verified, strict=True)):
            response = json.loads(task.finish(outcome))
            response["role_model_forwards"] = int(j == 0)
            if j == 0:
                response["role_group"] = "+".join(map(str, sorted(len(task.phrases) for _, task in deferred)))
            completed.append((i, json.dumps(response)))
    yield from completed


def resolve(service, arm, cutoff, lightweight, resident, clues, roles, vectors, *, withheld=None, inspect_source=None, source_guard=None):
    scores = {name: vectors[2:] @ service.banks[name].T for name in service.bands}
    ranking = service.index.full_rank(vectors[:2], clues, withheld)
    state = plan(service, ranking, scores, clues, roles, withheld, "screened", run_role=False)
    pair, base, response = state["pair"], state["roles"], state["response"]
    if arm == "matched_resident":
        if not pair and not response["calls"]:
            from experiments.credit.paired_role_resident_source import matched_source_proposal

            proposal = matched_source_proposal(service, ranking, scores, clues, resident)
            if not proposal:
                return json.dumps(payload(service.index, None, []))
            candidate, pools = proposal
            if inspect_source is not None:
                inspect_source(candidate, clues, ranking)
            if source_guard is not None and not source_guard(candidate, ranking):
                return json.dumps(payload(service.index, None, []))

            def finish(outcome):
                choices, gap, tokens = outcome
                accepted = cutoff is not None and gap > cutoff and None not in choices
                selected = [{role} for role in choices] if accepted else []
                return json.dumps(payload(service.index, candidate if accepted else None,
                                          selected, role_calls=1, tokens=tokens))

            return RoleCheck(roles, pools, finish)
        arm = "student_then_teacher"
    if arm == "dense_joint":
        if not pair and not response["calls"]:
            from experiments.credit.paired_role_joint_postings import dense_source_fallback

            return json.dumps(dense_source_fallback(service, ranking, scores, clues))
        arm = "student_then_teacher"
    if not pair or response["calls"] or response["owners"]:
        return json.dumps(response)
    if arm in ("reserve_agreement", "reserve_prescreen", *LIGHTWEIGHT_METHODS, *CASCADE_ARMS[2:]):
        scope = {"papers": pair, "roles": (None, None)}
        checked, _, slot, approved = route(service.index.articles, scope, scores, service.bands)
        if approved:
            def finish(outcome):
                (role,), gap, tokens = outcome
                accepted = role in approved and gap > service.cutoffs["role"]
                selected = replace(checked, slot, role) if accepted else checked
                return json.dumps(payload(service.index, pair, selected, role_calls=1, tokens=tokens))

            return RoleCheck([roles[slot]], [checked[slot]], finish)
        if candidates(service.index, pair, checked):
            return json.dumps(payload(service.index, pair, checked))
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
        def finish(outcome):
            choice, margin, tokens = outcome
            observation = dict(choice=choice if choice == eligible else None, margin=margin, forwards=1, tokens=tokens)
            return json.dumps(apply(service.index, pair, base, observation, cutoff))

        return RoleCheck(roles, pools, finish)
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
    if any(input_path is not None for input_path in (
            args.joint_postings_reference, args.joint_guard_reference, args.mixed_role_reference)):
        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, _, fit = fit_lightweight(args.plos, service, definitions, training)
        if args.mixed_role_reference is not None:
            from experiments.credit.paired_role_mixed_verifier import run_mixed

            run_mixed(service, labels, cutoff, (heads, definitions, fit),
                      args.mixed_role_reference, args.output, args.reverse)
        elif args.joint_guard_reference is not None:
            from experiments.credit.paired_role_joint_guard import run_guard

            run_guard(service, labels, cutoff, (heads, definitions, fit),
                      args.joint_guard_reference, args.output, args.reverse)
        else:
            from experiments.credit.paired_role_joint_postings import run_joint

            run_joint(service, labels, cutoff, heads, fit["reserve_student"]["cutoff"],
                      args.joint_postings_reference, args.output, args.reverse)
        return
    if args.soft_role_reference is not None:
        from experiments.credit.paired_role_soft_distillation import run_distillation

        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, _, fit = fit_lightweight(args.plos, service, definitions, training)
        run_distillation(service, labels, cutoff, (heads, definitions, fit), training, args.plos,
                         args.soft_role_reference, args.output, args.reverse)
        return
    if args.fused_4b_reference is not None:
        from experiments.credit.paired_role_fused_4b import run_fusion

        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, _, fit = fit_lightweight(args.plos, service, definitions, training)
        run_fusion(service, labels, cutoff, (heads, definitions, fit), args.plos,
                   args.fused_4b_reference, args.output, args.reverse)
        return
    if args.resident_source_reference is not None:
        from experiments.credit.paired_role_resident_source import run_resident

        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, _, fit = fit_lightweight(args.plos, service, definitions, training)
        matched = (args.matched_source_projection, args.matched_source_manifest,
                   args.matched_source_supplement, args.matched_source_supplement_manifest, wording)
        run_resident(service, labels, cutoff, (heads, definitions, fit), args.plos,
                     args.resident_source_reference, args.output, args.reverse,
                     matched=matched if args.matched_source_projection is not None else None)
        return
    by_arm, results = {}, {}
    resident = matched_serving = None
    lightweight = report = None
    requested = PRESCREEN_ARMS if args.prescreen else ARMS
    if args.plos:
        if digest(args.plos) != frozen["inputs"]["plos"]:
            raise ValueError("Changed PLOS calibration projection")
        heads, training_report, fit = fit_lightweight(args.plos, service, definitions, training)
        report = dict(training=training_report, calibration=fit, plos_sha256=digest(args.plos))
        lightweight = (heads, definitions, fit)
        matched_reference = args.matched_source_service_reference or args.fresh_source_reference
        if matched_reference is not None:
            from experiments.credit.paired_role_matched_source import calibrate_pairs
            from experiments.credit.paired_role_resident_source import calibrate, clue_view

            resident, _ = calibrate(service, args.plos)
            resident, matched_fit = calibrate_pairs(
                service, resident, args.matched_source_projection, args.matched_source_manifest,
                args.matched_source_supplement, args.matched_source_supplement_manifest, wording,
            )
            expected = json.loads(matched_reference.read_bytes())
            if matched_fit != expected["matched_calibration"]:
                raise ValueError("Service calibration differs from the frozen component test")
            agreement = Counter()
            for request, label in labels.items():
                clues = [part["topic"] for part in json.loads(request.payload)["requirements"]]
                agreement[label["view"]] += clue_view(clues) == label["view"]
            matched_serving = dict(calibration=matched_fit, observed_view_agreement=agreement,
                                   component_sha256=digest(matched_reference))
            if args.fresh_source_reference is not None:
                from experiments.credit.elife_fresh_source_evaluation import evaluate
                from experiments.credit.elife_extended_role_holdout import SALT

                if manifest.get("sampling_salt") != SALT:
                    raise ValueError("Expected the fixed extended eLife holdout")
                if args.source_audit is not None:
                    from experiments.credit.elife_source_identity_audit import evaluate
                evaluate(service, labels, cutoff, lightweight, resident, clusters(selected, articles),
                         args.output, args.reverse, dict(manifest_sha256=digest(args.manifest),
                                                       projection_sha256=digest(args.projection),
                                                       calibration=matched_serving, panel=census,
                                                       lightweight=report, gpu=torch.cuda.get_device_name(),
                                                       **revisions),
                         **({"prior": args.source_audit} if args.source_audit is not None else
                            {"guarded": True} if args.guarded_source_replay else {}))
                return
            requested = MATCHED_SERVICE_ARMS
            if args.embedding_batch_capacity or args.role_batch_capacity or args.mixed_role_batch_capacity:
                if args.embedding_batch_capacity and 4 * args.embedding_batch_capacity > service.encoder.batch_size:
                    raise ValueError("An embedding group must fit in a single encoder forward")
                requested = ("student_then_teacher", "matched_resident",
                             "student_then_teacher_batched", "matched_resident_batched")
        elif args.joint_postings_service:
            requested = JOINT_SERVICE_ARMS
        else:
            requested = CASCADE_ARMS if args.cascade else ("baseline", "reserve_prescreen", *LIGHTWEIGHT_METHODS)
        print("Frozen lightweight routes", fit, flush=True)
    batch_capacity = args.embedding_batch_capacity or args.role_batch_capacity or args.mixed_role_batch_capacity
    batch_stage = "embedding" if args.embedding_batch_capacity else "paired_role"
    if args.mixed_role_batch_capacity:
        batch_stage = "mixed_role"
    arms = requested[::-1] if args.reverse else requested
    for offset, rate in enumerate(RATES):
        trace, arrivals = arrivals_for(labels, rate, args.repeats)
        results[rate] = {}
        for arm in arms[offset:] + arms[:offset]:
            policy = arm.removesuffix("_batched")
            capacity = batch_capacity if policy != arm else 1
            group = None
            if capacity > 1:
                def group(requests):
                    return execute_group(service, requests, policy, cutoff, lightweight, resident, batch_stage)

            records, elapsed = replay(lambda request: execute(service, request, policy, cutoff, lightweight, resident),
                                      trace, arrivals, trace[:24], execute_group=group, capacity=capacity)
            detail, responses = summarize(trace, arrivals, records, elapsed, labels, args.repeats)
            if detail["failures"] or len(responses) != len(labels):
                raise ValueError(f"Incomplete replay for {arm} at {rate}: {detail['failures']}")
            calls = Counter(role_forwards=0, source_forwards=0)
            role_groups = Counter()
            for record in records:
                response = json.loads(record["response"])
                if response["role_calls"] + response["source_calls"] != response["calls"] or response["calls"] > 1:
                    raise ValueError("Shared verifier budget exceeded")
                calls.update(role_forwards=response["role_calls"], source_forwards=response["source_calls"])
                if args.role_batch_capacity or args.mixed_role_batch_capacity:
                    calls.update(role_model_forwards=response.get("role_model_forwards", response["role_calls"]))
                    if "role_group" in response:
                        role_groups[response["role_group"]] += 1
                if args.joint_postings_service or matched_serving is not None:
                    calls.update(source_rows=response.get("source_rows", response["source_calls"]))
            detail["verification"] = dict(calls)
            if args.mixed_role_batch_capacity:
                detail["shared_role_groups"] = dict(sorted(role_groups.items()))
            if batch_capacity:
                served = Counter(record["batch"] for record in records)
                if any(count % width for width, count in served.items()):
                    raise ValueError("Incomplete embedding group accounting")
                detail["request_groups"] = {width: count // width for width, count in served.items()}
                detail["embedding_forwards"] = (sum(detail["request_groups"].values())
                                                if batch_stage == "embedding" else len(records))
            if arm in by_arm and by_arm[arm] != responses:
                raise ValueError("Traffic changed the emitted evidence")
            by_arm[arm] = responses
            results[rate][arm] = detail
            print(rate, arm, detail["counts"], round(detail["p95_ms"]["total_ms"], 2), flush=True)
    if args.prescreen and not args.plos and by_arm["reserve_agreement"] != by_arm["reserve_prescreen"]:
        changed = sum(by_arm["reserve_agreement"][req] != by_arm["reserve_prescreen"][req] for req in labels)
        raise ValueError(f"Agreement prescreen changed {changed} evidence payloads")
    baseline = "student_then_teacher" if batch_capacity else "baseline"
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
            delta = [int(metrics[arm][req]["strict"]) - int(metrics[baseline][req]["strict"]) for req in positives]
            interval = clustered_paired_interval(delta, group_ids)
            interval["sampling_unit"] = "connected paper/recorded-author groups"
            contrasts[arm][view] = dict(totals=dict(counts), positives=len(positives),
                                        requests=len(rows), paired_strict_change=interval)
            if arm in ("dense_joint", "matched_resident", "matched_resident_batched"):
                extra = [int(metrics[arm][req]["strict"]) - int(metrics["student_then_teacher"][req]["strict"])
                         for req in positives]
                paired = clustered_paired_interval(extra, group_ids)
                paired["sampling_unit"] = "connected paper/recorded-author groups"
                contrasts[arm][view]["paired_strict_change_vs_cascade"] = paired
    signatures = {
        arm: sha256(json.dumps([proofs[req] for req in labels], ensure_ascii=False).encode()).hexdigest()
        for arm, proofs in by_arm.items()
    }
    if matched_serving is not None:
        expected_signatures = expected["signatures"]
        reference_arms = {"student_then_teacher": "cascade", "dense_joint": "dense_joint",
                          "matched_resident": "gold_matched"}
        for arm in requested:
            policy = arm.removesuffix("_batched")
            if policy not in reference_arms:
                continue
            if signatures[arm] != expected_signatures[reference_arms[policy]]:
                raise ValueError(f"Service evidence differs from the component result for {arm}")
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection),
        manifest_sha256=digest(args.manifest), paired_results_sha256=digest(args.paired_frozen),
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, prescreen=args.prescreen, census=census,
        distinct_requests=len(labels), author_connected_groups=len(set(groups.values())),
        cutoffs=frozen["cutoffs"] | {"two_batched": cutoff}, results=results, quality=contrasts,
        response_signatures=signatures, lightweight=report, cascade=args.cascade, **revisions,
        **({"joint_postings_service": True} if args.joint_postings_service else {}),
        **({"matched_source_service": matched_serving} if matched_serving is not None else {}),
        **({"batch_capacity": batch_capacity, "batch_stage": batch_stage} if batch_capacity else {}),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "paired_frozen", "niso", "training", "bands",
                 "source_frozen", "role_frozen", "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prescreen", action="store_true")
    parser.add_argument("--plos", type=Path)
    parser.add_argument("--cascade", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    batching = parser.add_mutually_exclusive_group()
    batching.add_argument("--embedding-batch-capacity", type=int, choices=range(2, 9))
    batching.add_argument("--role-batch-capacity", type=int, choices=range(2, 9))
    batching.add_argument("--mixed-role-batch-capacity", type=int, choices=range(2, 9))
    parser.add_argument("--matched-source-projection", type=Path)
    parser.add_argument("--matched-source-manifest", type=Path)
    parser.add_argument("--matched-source-supplement", type=Path)
    parser.add_argument("--matched-source-supplement-manifest", type=Path)
    parser.add_argument("--reverse", action="store_true")
    source_diagnostics = parser.add_mutually_exclusive_group()
    source_diagnostics.add_argument("--source-audit", type=Path)
    source_diagnostics.add_argument("--guarded-source-replay", action="store_true")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--stage-audit-reference", type=Path)
    diagnostics.add_argument("--source-postings-reference", type=Path)
    diagnostics.add_argument("--joint-postings-reference", type=Path)
    diagnostics.add_argument("--joint-postings-service", action="store_true")
    diagnostics.add_argument("--joint-guard-reference", type=Path)
    diagnostics.add_argument("--mixed-role-reference", type=Path)
    diagnostics.add_argument("--soft-role-reference", type=Path)
    diagnostics.add_argument("--fused-4b-reference", type=Path)
    diagnostics.add_argument("--resident-source-reference", type=Path)
    diagnostics.add_argument("--matched-source-service-reference", type=Path)
    diagnostics.add_argument("--fresh-source-reference", type=Path)
    args = parser.parse_args()
    if (args.embedding_batch_capacity or args.role_batch_capacity or args.mixed_role_batch_capacity) and args.matched_source_service_reference is None:
        parser.error("Batching requires --matched-source-service-reference")
    if (args.source_audit is not None or args.guarded_source_replay) and args.fresh_source_reference is None:
        parser.error("Source diagnostics require --fresh-source-reference")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.cascade and not args.plos:
        parser.error("--cascade requires --plos")
    if args.joint_postings_reference is not None and args.plos is None:
        parser.error("--joint-postings-reference requires --plos")
    if args.joint_postings_service and args.plos is None:
        parser.error("--joint-postings-service requires --plos")
    if args.joint_guard_reference is not None and args.plos is None:
        parser.error("--joint-guard-reference requires --plos")
    if args.mixed_role_reference is not None and args.plos is None:
        parser.error("--mixed-role-reference requires --plos")
    if args.soft_role_reference is not None and args.plos is None:
        parser.error("--soft-role-reference requires --plos")
    if args.fused_4b_reference is not None and args.plos is None:
        parser.error("--fused-4b-reference requires --plos")
    if args.resident_source_reference is not None and args.plos is None:
        parser.error("--resident-source-reference requires --plos")
    matched_inputs = (args.matched_source_projection, args.matched_source_manifest,
                      args.matched_source_supplement, args.matched_source_supplement_manifest)
    reference = args.resident_source_reference or args.matched_source_service_reference or args.fresh_source_reference
    if any(value is not None for value in matched_inputs) or args.matched_source_service_reference or args.fresh_source_reference:
        if (reference is None
                or any(value is None for value in matched_inputs) or args.plos is None):
            parser.error("Matched source calibration requires the resident reference and both complete snapshots")
    run(args)
