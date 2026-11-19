"""Test prefix continuation and one-pass evidence routing on MERIT."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.decision.merit_evidence_dispatch import Decision, SALT, SYSTEM, decide, read_cases, terms
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("two", "full", "rerun", "reuse", "reuse_guarded", "upfront")
ESCALATION_BUDGET = 0.25
GUARD = 0.5


def prompt_ids(reader, case, count):
    publications = "\n\n".join(f"{i}. {text}" for i, text in enumerate(case.papers[:count], 1))
    messages = [dict(role="system", content=SYSTEM),
                dict(role="user", content=case.target + "\n\nREVIEWER PUBLICATIONS\n" + publications)]
    return reader.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)


def shared_length(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class PrefixDecision(Decision):
    @torch.inference_mode()
    def assess(self, ids, *, persist=False, cache=None):
        token_array = torch.tensor([ids], device="cuda")
        output = self.model(input_ids=token_array, use_cache=persist, past_key_values=cache, logits_to_keep=1)
        logits = output.logits[0, -1, self.labels].float().cpu().tolist()
        return logits[1] - logits[0], output.past_key_values


def before_reading(case):
    target_terms = terms(case.query)
    texts = [terms(p) for p in case.papers]
    first = set.union(*texts[:2])
    tail_words = set.union(set(), *texts[2:])
    denom = max(1, len(target_terms))
    coverage = len(target_terms & first) / denom
    additional = len((target_terms & tail_words) - first) / denom
    return (coverage, additional, len(case.papers), len(case.target),
            sum(len(p) for p in case.papers[2:]), coverage * additional)


def observe(reader, case, depth):
    ids = prompt_ids(reader, case, depth)
    margin, _ = reader.assess(ids)
    return dict(margin=margin, tokens=len(ids))


def fit(rows):
    features = np.array([before_reading(case) for case, _, _ in rows])
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-9)
    normalized = (features - center) / scale
    target = np.array([int(decide(full) == case.label) - int(decide(short) == case.label)
                       for case, short, full in rows])
    weights = np.linalg.solve(normalized.T @ normalized + 16 * np.eye(normalized.shape[1]),
                              normalized.T @ (target - target.mean()))
    return center, scale, weights, float(target.mean())


def score_upfront(case, model):
    center, scale, weights, intercept = model
    gain = (np.asarray(before_reading(case)) - center) / scale @ weights + intercept
    return float(gain * 10000 / max(1000, sum(len(text) for text in case.papers[2:])))


def choose_threshold(rows, *, upfront, model):
    pred = [score_upfront(c, model) if upfront else -abs(s["margin"]) for c, s, _ in rows]
    positives = sum(c.label for c, _, _ in rows)
    negatives = len(rows) - positives
    budget = ESCALATION_BUDGET * sum(f["tokens"] for _, _, f in rows)
    options = []
    for cutoff in sorted(set(pred) | {max(pred) + 1}):
        cost = 0
        utility = 0.
        for rank, (case, short, full) in zip(pred, rows, strict=True):
            expand = rank >= cutoff and len(case.papers) > 2
            if expand:
                cost += full["tokens"] - (short["tokens"] if upfront else 0)
            utility += (decide(full if expand else short) == case.label) / (positives if case.label else negatives)
        if cost <= budget:
            options.append((utility / 2, -cost, cutoff))
    best = max(options) if upfront else max(options, key=lambda item: (-item[1], item[0], item[2]))
    return dict(cutoff=best[2], calibration_balanced_accuracy=best[0],
                calibration_incremental_tokens=-best[1], token_budget=budget)


def run_arm(reader, case, arm, gates, model):
    started = perf_counter()
    reused = 0
    rechecked = 0
    expand = arm == "full" or (arm == "upfront" and len(case.papers) > 2
                               and score_upfront(case, model) >= gates["upfront"]["cutoff"])
    persist = len(case.papers) > 2 and arm in ("reuse", "reuse_guarded")
    ids = prompt_ids(reader, case, len(case.papers) if expand else 2)
    margin, past = reader.assess(ids, persist=persist)
    tokens = len(ids)
    escalated = arm == "upfront" and expand
    if arm in ("rerun", "reuse", "reuse_guarded") and len(case.papers) > 2:
        if -abs(margin) >= gates["sequential"]["cutoff"]:
            escalated = True
            full_ids = prompt_ids(reader, case, len(case.papers))
            if persist:
                reused = min(shared_length(ids, full_ids), len(full_ids) - 1)
                if not reused or past.get_seq_length() != len(ids):
                    raise ValueError("Unexpected empty or incomplete prefix cache")
                remove = len(ids) - reused
                if remove:
                    past.crop(-remove)
                if past.get_seq_length() != reused:
                    raise ValueError("Cache crop did not retain the exact shared prefix")
                margin, _ = reader.assess(full_ids[reused:], persist=True, cache=past)
            else:
                margin, _ = reader.assess(full_ids)
            tokens += len(full_ids) - reused
            if arm == "reuse_guarded" and abs(margin) <= GUARD:
                margin, _ = reader.assess(full_ids)
                rechecked = 1
                tokens += len(full_ids)
    return dict(margin=margin, tokens=tokens, reused=reused, rechecked=rechecked,
                escalated=escalated, ms=(perf_counter() - started) * 1000)


def summarize(rows):
    out = {}
    groups = [case.cluster for case, _ in rows]
    for arm in ARMS:
        values = [result[arm] for _, result in rows]
        gold = [c.label for c, _ in rows]
        predicted = [decide(v) for v in values]
        accuracy = [a == b for a, b in zip(predicted, gold, strict=True)]
        tp = sum(a == b == 1 for a, b in zip(predicted, gold, strict=True))
        tn = sum(a == b == 0 for a, b in zip(predicted, gold, strict=True))
        n_positive = sum(gold)
        fp = len(gold) - n_positive - tn
        fn = n_positive - tp
        full = [decide(r["full"]) == c.label for c, r in rows]
        ref = [r["rerun"] for _, r in rows]
        out[arm] = dict(
            correct=sum(accuracy), accuracy=mean(accuracy),
            balanced_accuracy=(tp/n_positive + tn/(len(gold)-n_positive))/2,
            f1=2*tp/max(1,2*tp+fp+fn), tp=tp, fp=fp, fn=fn, tn=tn,
            delta_vs_full=clustered_paired_interval([int(a)-int(b) for a,b in zip(accuracy,full,strict=True)], groups),
            mean_ms=mean(v["ms"] for v in values),
            p50_p95_ms=np.quantile([v["ms"] for v in values], [.5, .95]).tolist(),
            mean_tokens=mean(v["tokens"] for v in values), escalated=sum(v["escalated"] for v in values),
            rechecked=sum(v["rechecked"] for v in values),
            max_reused=max(v["reused"] for v in values),
            decision_diffs_vs_rerun=sum(decide(a)!=decide(b) for a,b in zip(values,ref,strict=True)),
            escalation_diffs_vs_rerun=sum(a["escalated"]!=b["escalated"] for a,b in zip(values,ref,strict=True)),
            mean_ms_escalated=mean(v["ms"] for v in values if v["escalated"]) if any(v["escalated"] for v in values) else None,
        )
    return out


def run(data, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a new output path and Slurm CUDA")
    splits = dict(fit=[], calibration=[], test=[])
    for case in read_cases(data):
        group = int(sha256(f"{SALT}:{case.cluster}".encode()).hexdigest()[:8], 16) % 10
        splits["fit" if group<4 else "calibration" if group<6 else "test"].append(case)
    reader = PrefixDecision(checkpoint)
    observe(reader, splits["fit"][0], 2)
    warm = next(case for case in splits["fit"] if len(case.papers) > 2)
    forced = {"sequential": {"cutoff": float("-inf")}}
    for arm in ("rerun", "reuse", "reuse_guarded"):
        run_arm(reader, warm, arm, forced, None)
    pairs = {}
    for name in ("fit", "calibration"):
        pairs[name] = [(c, observe(reader,c,2), observe(reader,c,len(c.papers))) for c in splits[name]]
        print(f"Completed {name}: {len(pairs[name])}", flush=True)
    model = fit(pairs["fit"])
    gates = {name: choose_threshold(pairs["calibration"], upfront=name=="upfront", model=model)
             for name in ("sequential","upfront")}
    print("Frozen calibration:", gates, flush=True)
    test = []
    for i, case in enumerate(splits["test"]):
        first = i % len(ARMS)
        test.append((case, {arm: run_arm(reader,case,arm,gates,model) for arm in ARMS[first:]+ARMS[:first]}))
        if (i+1) % 40 == 0:
            print(f"Completed test: {i+1}", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=reader.provenance,
        data_sha256=digest(data), script_sha256=digest(Path(__file__)),
        gates=gates, guard_logit=GUARD, split_counts={k:len(v) for k,v in splits.items()},
        test_groups=len({c.cluster for c in splits["test"]}), arms=summarize(test),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data","checkpoint","output"):
        parser.add_argument(name,type=Path)
    run(**vars(parser.parse_args()))
