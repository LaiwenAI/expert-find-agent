"""Score two independently attributable relations in one reader prompt."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import conjunctive_proof_injection as injection
from experiments.docred import conjunctive_proof_windows as proof
from experiments.docred import oracle_role_evidence as readers
from experiments.docred import person_bound_evidence as names
from experiments.docred import work_owner_contrast as owners
from experiments.followir.core17_reranking import digest, write_json


LIVE = ("isolated", "binary_shared", "joint", "joint_symmetric")
ARMS = (*LIVE, "joint_event")
SCOPE = (
    "Evaluate whether the numbered source sentences explicitly support credit for the "
    "named person in the requested role on each work. Use only the sentence IDs listed "
    "for each work. Attribution to someone else or to a different work is insufficient. "
)
BINARY = SCOPE + "Judge only the specified work. Reply 1 for explicit support or 0 otherwise."
JOINT = SCOPE + "The two claims are independent. Reply 0 for neither, 1 for the first work only, " \
                "2 for the second work only, or 3 for both."


def shared_context(case, windows, positions):
    used = sorted({i for p in positions for i in windows[p]})
    records = {str(i + 1): case["sentences"][i] for i in used}
    first, second = (case["names"][i] for i in positions)
    if first[1] != second[1]:
        raise ValueError("Joint scope must remain on one person")
    return dict(
        person=first[1], role=names.ROLES[case["role"]], sentences=records,
        works=[dict(name=work, allowed_sentences=[str(j+1) for j in windows[i]])
               for i, (work, _) in zip(positions,(first,second),strict=True)],
    )


def messages(case, windows, arm):
    if arm == "isolated":
        return [(owners.SYSTEM, f"Source:\n{doc}\n\nQuestion: {query}")
                for query, doc in injection.prompts(case, windows)]
    if arm == "binary_shared":
        prompts = []
        for positions in proof.PAIR:
            data = shared_context(case, windows, positions)
            prompts.extend((BINARY, json.dumps(dict(data, judge_work=i+1), ensure_ascii=False)) for i in range(2))
        return prompts
    prompts = [(JOINT, json.dumps(shared_context(case, windows, positions), ensure_ascii=False))
               for positions in proof.PAIR]
    if arm == "joint_symmetric":
        prompts.extend((JOINT, json.dumps(shared_context(case, windows, positions[::-1]), ensure_ascii=False))
                       for positions in proof.PAIR)
    return prompts


def bit_margins(scores):
    first = np.logaddexp(scores[:,1], scores[:,3]) - np.logaddexp(scores[:,0], scores[:,2])
    second = np.logaddexp(scores[:,2], scores[:,3]) - np.logaddexp(scores[:,0], scores[:,1])
    return np.column_stack((first,second))


def score_model(case, windows, tokenizer, model, token_ids, arm):
    import torch

    start = perf_counter()
    rendered = [tokenizer.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True, tokenize=True, return_dict=False,
    ) for system,user in messages(case, windows, arm)]
    if max(map(len, rendered)) > 2048:
        raise ValueError("Exceeded the fixed maximum prompt length")
    batch = tokenizer.pad([{"input_ids": ids} for ids in rendered], padding=True,
                          padding_side="left", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, token_ids].float().cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite decisions")
    event = None
    disagreement = 0
    if arm == "isolated":
        values = (logits[:,1] - logits[:,0])[list(np.ravel(proof.PAIR))].reshape(2,2)
    elif arm == "binary_shared":
        values = (logits[:,1] - logits[:,0]).reshape(2,2)
    else:
        values = bit_margins(logits)
        if arm == "joint":
            event = logits[:,3] - np.logaddexp.reduce(logits[:,:3],axis=1)
        else:
            reversed_values = values[2:, ::-1]
            disagreement = int(np.count_nonzero((values[:2] > 0) != (reversed_values > 0)))
            values = np.minimum(values[:2], reversed_values)
    return dict(
        values=values.min(axis=1),
        atomic=values, event=event, order_disagreement=disagreement,
        tokens=batch["input_ids"].numel(), prompts=len(rendered),
        elapsed=(perf_counter()-start)*1000,
    )


def collect(panel, tokenizer, model, token_ids, offset):
    results = []
    for i, case in enumerate(panel):
        before = perf_counter()
        windows, _ = injection.baseline(case)
        prep = (perf_counter() - before) * 1000
        order = (i + offset) % len(LIVE)
        chosen = {}
        for arm in LIVE[order:]+LIVE[:order]:
            observation = score_model(case, windows, tokenizer, model, token_ids, arm)
            observation["elapsed"] += prep
            chosen[arm] = observation
        chosen["joint_event"] = dict(chosen["joint"], values=chosen["joint"]["event"])
        results.append((case,chosen))
        if (i+1) % 60 == 0:
            print(f"{i+1}/{len(panel)}", flush=True)
    return results


def fit(rows):
    cutoffs={}
    for arm in ARMS:
        unjudged = sorted((v for case, scores in rows
                           for v, known in zip(scores[arm]["values"], case["known"], strict=True)
                           if not known), reverse=True)
        cutoffs[arm] = float(unjudged[int(len(unjudged)*proof.DEV_RISK)])
    return cutoffs


def summarize(rows, cutoffs):
    metrics = {}
    groups = [case["doc"] for case,_ in rows]
    reference = np.array([
        np.count_nonzero((observations["isolated"]["values"] > cutoffs["isolated"]) & case["known"])
        for case, observations in rows
    ])
    for arm, limit in cutoffs.items():
        accepted_known, times = [], []
        totals = Counter()
        for case, observations in rows:
            obs=observations[arm]
            decisions = obs["values"] > limit
            positives = np.asarray(case["known"])
            value = int(np.count_nonzero(decisions & positives))
            accepted_known.append(value)
            times.append(obs["elapsed"])
            underlying = np.asarray([case["labeled"][i] for i in np.ravel(proof.PAIR)]).reshape(2,2)
            totals.update(
                accepted_known=value, known_candidates=int(positives.sum()),
                accepted_unjudged=int(np.count_nonzero(decisions & ~positives)),
                unjudged_candidates=int((~positives).sum()),
                atomic_known_above_zero=int(np.count_nonzero((obs["atomic"] > 0) & underlying)),
                atomic_order_disagreements=obs["order_disagreement"],
                tokens=obs["tokens"], prompts=obs["prompts"],
            )
        deltas = (np.asarray(accepted_known) - reference) / 2
        metrics[arm] = dict(**totals, cutoff=limit, p95_ms=float(np.quantile(times,.95)),
                            paired_known_delta_vs_isolated=clustered_paired_interval(deltas,groups))
    return dict(cases=len(rows), documents=len(set(groups)), arms=metrics)


def run(release, checkpoint, output, offset):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pools, pins = proof.load(release)
    tokenizer, model, provenance = readers.load_reader(checkpoint)
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(4)]
    if any(len(ids) != 1 for ids in digits) or len({ids[0] for ids in digits}) != 4:
        raise ValueError("Expected four distinct one-token choices")
    token_ids = [ids[0] for ids in digits]
    collect(pools["calibration"][:1], tokenizer, model, token_ids, offset)
    measurements = {group: collect(panel, tokenizer, model, token_ids, offset) for group, panel in pools.items()}
    cutoffs = fit(measurements["calibration"])
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), source_hashes=pins,
        reader_revision=provenance["revision"], order_offset=offset,
        script_sha256=digest(Path(__file__)), window_builder_sha256=digest(Path(proof.__file__)),
        panels={name: summarize(rows, cutoffs) for name, rows in measurements.items()},
    ))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("release", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args=parser.parse_args()
    run(args.release,args.checkpoint,args.output,args.order_offset)
