"""Replace a proof-gated reader call with one candidate-masked linear head."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_one_role_transfer import build
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_joint_calibration import SUPPORT, judge, open_calibration
from experiments.credit.engineering_one_role_recheck import RoleReader, combined, pick, resolve, state
from experiments.credit.engineering_role_banks import TASK, encode, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


NONE = len(ROLES)
STUDENTS = ("dense_agreement", "label_head", "distilled_head")


def options(values, pool):
    indices = sorted(SUPPORT[role] for role in pool) + [NONE]
    best, runner_up = sorted(indices, key=lambda i: (-values[i], i))[:2]
    return (ROLES[best] if best != NONE else None), float(values[best] - values[runner_up])


def fit_heads(text, vectors, bank, band, teacher):
    examples = []
    for i, phrase in enumerate(text):
        gold = ROLES[i // 6]
        margin = vectors[i] @ bank.T
        ranked = [ROLES[j] for j in np.argsort(-margin, kind="stable")]
        original = {role for role, score in zip(ROLES, margin, strict=True) if margin.max() - score <= band}
        for pool in (set(ROLES), original | {gold, ranked[1]}, set([r for r in ranked if r != gold][:3])):
            prediction, _, _ = teacher.read(phrase, pool)
            examples.append((i, sorted(SUPPORT[r] for r in pool) + [NONE],
                             SUPPORT[gold] if gold in pool else NONE,
                             SUPPORT[prediction] if prediction else NONE))
    x = torch.as_tensor(np.asarray([vectors[i] for i, _, _, _ in examples]), device="cuda")
    label = F.one_hot(torch.as_tensor([y for _, _, y, _ in examples], device="cuda"), NONE + 1).float()
    teacher_labels = F.one_hot(torch.as_tensor([y for _, _, _, y in examples], device="cuda"), NONE + 1).float()
    masks = torch.zeros((len(examples), NONE + 1), dtype=torch.bool, device="cuda")
    for i, (_, allowed, _, _) in enumerate(examples):
        masks[i, allowed] = True
    weights, histories = {}, {}
    for name, mix in (("label_head", 0.), ("distilled_head", .5)):
        head = nn.Linear(x.shape[1], NONE + 1, device="cuda")
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        targets = (1. - mix) * label + mix * teacher_labels
        optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.03)
        for _ in range(180):
            selected = head(x).masked_fill(~masks, -1e4)
            loss = -(F.log_softmax(selected, dim=1) * targets).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        weights[name] = (head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy())
        histories[name] = dict(train_loss=float(loss.detach()), parameters=sum(p.numel() for p in head.parameters()))
    return weights, dict(examples=len(examples), teacher_matches_gold=int((label.argmax(1) == teacher_labels.argmax(1)).sum()),
                         training=histories)


def student_values(arm, vector, scores, heads, slot):
    if arm == "dense_agreement":
        values = [scores[key][slot] for key in ("ridge", "prototype")]
        standard = [(v - v.max()) / max(.001, float(v.std())) for v in values]
        return np.append(np.mean(standard, axis=0), -20.)
    weight, bias = heads[arm]
    return weight @ vector + bias


def route(articles, row, scores, bands):
    sets, initial, branches = state(articles, row, scores["ridge"], bands)
    slot = pick(branches, scores["ridge"])["proof_aware_gated"]
    if slot is None:
        return sets, initial, slot, set()
    prototype = required_roles(row, scores, "prototype_band", bands)
    approved = {role for role in sets[slot] & prototype[slot] if combined(articles, row, sets, slot, role)}
    return sets, initial, slot, approved


def calibrate(articles, cases, embeddings, banks, bands, heads):
    observations = defaultdict(list)
    for row in cases:
        vectors = embeddings[[SUPPORT[role] for role in row["roles"]]]
        scores = {name: vectors @ banks[name].T for name in bands}
        sets, initial, slot, approved = route(articles, row, scores, bands)
        baseline = judge(row, initial, 0.)
        for arm in STUDENTS:
            confidence, report = 0., baseline
            if approved:
                role, confidence = options(student_values(arm, vectors[slot], scores, heads, slot), sets[slot])
                if role in approved:
                    report = judge(row, combined(articles, row, sets, slot, role), 0.)
            observations[arm].append((confidence, report, baseline))
    fits = {}
    for arm, rows in observations.items():
        best = None
        for cutoff in sorted({0., *(gap for gap, _, _ in rows)}):
            tally = Counter()
            for gap, report, baseline in rows:
                tally.update(report if gap > cutoff else baseline)
            if tally["adverse"] <= len(rows) // 20:
                key = tally["exact_proof"], -tally["adverse"], cutoff
                if best is None or key > best[0]:
                    best = key, dict(cutoff=cutoff, counts=tally)
        fits[arm] = best[1] if best else None
    return fits


def evaluate(articles, rows, encoder, banks, wording, teacher, teacher_cutoff, bands, heads, fit, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, times = defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        intended = tuple(row["papers"])
        excerpts = [articles[paper]["clue"] for paper in intended]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(clue) for clue in excerpts])):
            start = perf_counter()
            prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            prompts += [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in text]
            vectors = encoder.encode(prompts)
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            embed_ms = (perf_counter() - start) * 1000
            teacher_cache = {}
            for availability, masked in zip(AVAILABILITY, (None, *intended), strict=True):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], clues, masked)
                pair, _ = select(catalog, topic, row, [set(), set()], "anchored", masked)
                case = row | {"papers": pair, "gold": row["gold"] if pair == intended else []}
                retrieve_ms = embed_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                baseline = resolve(articles, case, required_roles(case, scores, "prototype_band", bands)) if pair else {}
                results = {"prototype_band": (baseline, 0, 0., (perf_counter() - start) * 1000)}
                start = perf_counter()
                sets, initial, slot, approved = route(articles, case, scores, bands) if pair else ([], {}, None, set())
                guard_ms = (perf_counter() - start) * 1000
                if approved and slot not in teacher_cache:
                    start = perf_counter()
                    answer = teacher.read(text[slot], sets[slot])
                    teacher_cache[slot] = answer, (perf_counter() - start) * 1000
                for arm in ("teacher", *STUDENTS):
                    start = perf_counter()
                    output, called, read_ms, tokens = initial, 0, 0., 0
                    if approved:
                        if arm == "teacher":
                            (role, confidence, tokens), read_ms = teacher_cache[slot]
                            cutoff = teacher_cutoff
                            called = 1
                        else:
                            role, confidence = options(
                                student_values(arm, vectors[2 + slot], scores, heads, slot), sets[slot])
                            cutoff = fit[arm]["cutoff"] if fit[arm] else float("inf")
                        if role in approved and confidence > cutoff:
                            output = combined(articles, case, sets, slot, role)
                    results[arm] = output, called, tokens, guard_ms + read_ms + (perf_counter() - start) * 1000
                for arm, (output, calls, tokens, ms) in results.items():
                    start = perf_counter()
                    observed = judge(case, output, 0.)
                    json.dumps(observed)
                    if i:
                        key = "|".join((view, availability, arm))
                        counts[key].update(observed, cases=1, calls=calls, tokens=tokens,
                                           substitutions=observed["issued"] and pair != intended,
                                           swap_issued=row["variant"] == "swap" and observed["issued"])
                        times[key].append(retrieve_ms + ms + (perf_counter() - start) * 1000)
    return {key: dict(totals=val, mean_ms=float(np.mean(times[key])), p95_ms=float(np.quantile(times[key], .95)))
            for key, val in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    teacher = RoleReader(args.reader, definitions)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    heads, training_summary = fit_heads(training, encode(encoder, training), banks["ridge"], bands["ridge"], teacher)
    plos, _, cases = open_calibration(args.plos)
    fit = calibrate(plos, cases, encode(encoder, VIEWS["original"]), banks, bands, heads)
    articles, panel, census = build(args.elife)
    engineering = json.loads(args.engineering.read_bytes())
    cohorts = {"elife": (articles, panel), "engineering": (engineering["articles"], engineering["queries"])}
    results = {name: evaluate(*data, encoder, banks, wording, teacher, cutoff, bands, heads, fit, args.reverse)
               for name, data in cohorts.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), elife_sha256=digest(args.elife), engineering_sha256=digest(args.engineering),
        calibration_sha256=digest(args.plos), elife_panel=census, calibration_cases=len(cases),
        embedding_revision=checkpoint["revision"], reader_revision=teacher.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, training=training_summary, fit=fit, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("elife", "engineering", "plos", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
