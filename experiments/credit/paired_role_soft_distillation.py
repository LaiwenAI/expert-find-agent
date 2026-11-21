"""Fit role heads to the same hard or soft 4B preferences over hard negatives."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.credit.distilled_role_recheck import NONE, options
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_joint_calibration import SUPPORT, open_calibration
from experiments.credit.engineering_role_banks import encode
from experiments.credit.paired_role_mixed_verifier import availability
from experiments.credit.paired_role_stage_audit import signatures
from experiments.credit.paired_role_verification import alternative_roles
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_confirmation_transfer import verify
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.followir.core17_reranking import digest, write_json


STUDENTS = ("hard", "soft")
ARMS = ("cascade", "dense_joint", *STUDENTS)


def train(service, phrases):
    vectors = encode(service.encoder, phrases)
    shortlist = alternative_roles({name: vectors @ bank.T for name, bank in service.banks.items()})
    prompts, choices, x, labels = [], [], [], []
    for i, (phrase, leading) in enumerate(zip(phrases, shortlist, strict=True)):
        gold = ROLES[i // 6]
        rival_order = np.argsort(-service.banks["ridge"].dot(vectors[i]), kind="stable")
        nongold = [ROLES[index] for index in rival_order if ROLES[index] != gold][:3]
        pools = tuple(dict.fromkeys((
            frozenset(ROLES), frozenset(leading), frozenset([gold, *nongold[:2]]), frozenset(nongold),
        )))
        rendered, options_by_prompt = service.role.format_batched([phrase] * len(pools), pools)
        prompts.extend(rendered)
        choices.extend(options_by_prompt)
        x.extend([vectors[i]] * len(pools))
        labels.extend(SUPPORT[gold] if gold in pool else NONE for pool in pools)
    observations = []
    for offset in range(0, len(prompts), 24):
        scores, _ = service.role.option_logits(prompts[offset:offset + 24])
        observations.extend(scores)
    target = np.full((len(prompts), NONE + 1), -1e4, dtype=np.float32)
    mask = np.zeros(target.shape, dtype=bool)
    for row, (allowed, teacher) in enumerate(zip(choices, observations, strict=True)):
        indices = [SUPPORT[role] if role is not None else NONE for role in allowed]
        target[row, indices] = teacher[:len(indices)]
        mask[row, indices] = True
    x = torch.as_tensor(np.asarray(x), device="cuda")
    y = F.one_hot(torch.as_tensor(labels, device="cuda"), NONE + 1).float()
    logits = torch.as_tensor(target, device="cuda")
    allowed = torch.as_tensor(mask, device="cuda")
    teacher_hard = F.one_hot(logits.argmax(dim=1), NONE + 1).float()
    teacher_soft = F.softmax(logits / 2., dim=1)
    result, losses = {}, {}
    for name, supervision in (("hard", teacher_hard), ("soft", teacher_soft)):
        head = nn.Linear(x.shape[1], NONE + 1, device="cuda")
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.03)
        for _ in range(180):
            prediction = F.log_softmax(head(x).masked_fill(~allowed, -1e4), dim=1)
            loss = -(prediction * (.5 * y + .5 * supervision)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        result[name] = (head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy())
        losses[name] = float(loss.detach())
    return result, dict(phrases=len(phrases), masked_examples=len(prompts),
                        teacher_correct=int((logits.argmax(dim=1) == y.argmax(dim=1)).sum().item()),
                        loss=losses, training_device=torch.cuda.get_device_name())


def predict(head, vectors, pools):
    weights, bias = head
    selected, gaps = zip(*(options(weights @ vector + bias, pool)
                            for vector, pool in zip(vectors, pools, strict=True)), strict=True)
    required = [{role} for role in selected] if None not in selected else None
    return required, min(gaps)


def fit_cutoffs(service, heads, plos):
    articles, _, cases = open_calibration(plos)
    catalog = SimpleNamespace(articles=articles)
    vectors = encode(service.encoder, VIEWS["original"])
    cases_by_student = defaultdict(list)
    for label in cases:
        subset = vectors[[SUPPORT[role] for role in label["roles"]]]
        pools = alternative_roles({name: subset @ bank.T for name, bank in service.banks.items()})
        for name, head in heads.items():
            required, gap = predict(head, subset, pools)
            if required is not None:
                result = evidence(payload(catalog, label["papers"], required), label)
            else:
                result = evidence(payload(catalog, None, []), label)
            cases_by_student[name].append((gap, result))
    calibrated = {}
    for name, samples in cases_by_student.items():
        combinations = []
        for cutoff in sorted({0., *(gap for gap, _ in samples)}):
            accepted = [result for gap, result in samples if gap > cutoff]
            strict = sum(row["strict"] for row in accepted)
            adverse = sum(row["adverse"] for row in accepted)
            if adverse == 0:
                combinations.append((strict, cutoff))
        strict, cutoff = max(combinations)
        calibrated[name] = dict(strict=strict, adverse=0, cutoff=cutoff, requests=len(samples))
    return calibrated


def run_distillation(service, labels, cutoff, lightweight, training, plos, reference, output, reverse):
    heads, train_report = train(service, training)
    cuts = fit_cutoffs(service, heads, plos)
    requests = list(labels)
    totals = defaultdict(lambda: defaultdict(Counter))
    outputs = defaultdict(dict)
    fallback_diagnostics = defaultdict(lambda: defaultdict(Counter))
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        view = label["view"]
        cascade, plan = availability(service, request, cutoff, lightweight)
        outcomes = dict.fromkeys(ARMS, cascade)
        if plan is not None:
            pair, pools, dense_roles, clues, _, resident_vectors, _ = plan
            required = {"dense_joint": dense_roles}
            for method in STUDENTS:
                proposed, margin = predict(heads[method], resident_vectors, pools)
                required[method] = proposed if margin > cuts[method]["cutoff"] else None
            compatible = {method: value for method, value in required.items()
                          if value is not None and payload(service.index, pair, value)["owners"]}
            accepted = verify(service.source, service.index, clues, pair, (0, 1)) > service.cutoffs["source"]
            for method, description in compatible.items():
                outcomes[method] = payload(service.index, pair if accepted else None, description, source_calls=1)
            for method in STUDENTS:
                fallback_diagnostics[view][method].update(
                    evaluated=1, role_prediction=required[method] is not None, source_called=method in compatible
                )
        for method, result in outcomes.items():
            if result["calls"] > 1 or result["calls"] != result["source_calls"] + result["role_calls"]:
                raise ValueError("Policy exceeded its verifier budget")
            totals[view][method].update(evidence(result, label))
            outputs[method][request] = result
    proofs = signatures(outputs, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]
    if proofs["cascade"] != expected["student_then_teacher"] or proofs["dense_joint"] != expected["dense_joint"]:
        raise ValueError("Policies do not reproduce the archived service")
    write_json(output, dict(requests=len(requests), counts=totals, signatures=proofs, calibration=cuts,
                            training=train_report, fallback_diagnostics=fallback_diagnostics,
                            reverse=reverse, reference_sha256=digest(reference),
                            script_sha256=digest(Path(__file__))))
