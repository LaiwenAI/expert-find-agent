"""Test candidate-depth, identity and source recovery at equal reader-pair count."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import nomination_evidence_windows as windows
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence

ARMS = ("single_batch", "staged_depth", "static_mix", "state_gap")
FIRST = 3


def recovery_pool(query, sources, profiles, documents, spans, postings, initial):
    seen = set(initial)
    alternate_docs = [d for d in routing.bm25_scores(query, postings) if d not in sources]
    expanded = {}
    for doc in sources:
        _, work, cues = profiles[doc]
        names = [(name, tuple(sorted(positions))) for name, positions, _ in
                 proposal.pool(documents[doc], query, profiles[doc], spans[doc]) if (doc, name) not in seen]
        expanded[doc] = names, work, cues
    return dict(
        depth=initial[FIRST:],
        identity=index.rank_people(query, sources, expanded)[0],
        source=index.rank_people(query, alternate_docs, profiles)[0],
    )


def pick(pool, order, cap):
    selected = []
    seen = set()
    depth = 0
    while len(selected) < cap and any(len(pool[k]) > depth for k in order):
        for kind in order:
            if depth < len(pool[kind]):
                pair = pool[kind][depth]
                if pair not in seen:
                    seen.add(pair)
                    selected.append((kind, pair))
                    if len(selected) == cap:
                        break
        depth += 1
    return selected


def score(query, pairs, read, context):
    prompts, indices = [], []
    work = " ".join(query["work"])
    role = evidence.ROLES[query["role"]]
    case = dict(work=work, role=query["role"])
    for doc, person in pairs:
        name = " ".join(person)
        text, parsed = context[doc]
        selected = evidence.choose(case, name, parsed, "lexical_two")
        indices.append(tuple(selected))
        prompts.append((f"Was {name} credited in the role of {role} for {work}?",
                        "\n".join(text[i] for i in selected)))
    margins, clipped = read(prompts) if prompts else ([], 0)
    if clipped or not np.isfinite(margins).all():
        raise ValueError("Invalid model output")
    return list(zip(pairs, indices, margins, strict=True))


def run_arm(query, arm, lookup, postings, profiles, documents, spans, read, context):
    started = perf_counter()
    docs, _, uncertain = routing.route(query, postings, lookup, "swap_then_guard")
    initial = index.rank_people(query, docs, profiles)[0]
    if arm == "single_batch":
        chosen = score(query, initial, read, context)
        phases, steps = [], int(bool(chosen))
    else:
        first = initial[:FIRST]
        chosen = score(query, first, read, context)
        steps = int(bool(first))
        remaining = len(initial) - len(first)
        phases = []
        if remaining:
            if arm == "staged_depth":
                actions = [("depth", pair) for pair in initial[FIRST:]]
            else:
                pool = recovery_pool(query, docs, profiles, documents, spans, postings, initial)
                supported = any(margin > auto.CUTOFF for _, _, margin in chosen)
                order = ("depth", "source", "identity")
                if arm == "state_gap":
                    order = ("depth", "identity", "source") if supported else (
                        ("source", "identity", "depth") if uncertain else ("identity", "depth", "source")
                    )
                actions = pick(pool, order, remaining)
            new = score(query, [pair for _, pair in actions], read, context)
            chosen += new
            steps += int(bool(new))
            phases = [kind for kind, _ in actions]
    if len(chosen) != len(initial):
        raise ValueError("A method used a different verification budget")
    elapsed = (perf_counter() - started) * 1000
    return chosen, phases, steps, elapsed


def assess(query, observations, gold, identities):
    chosen, phases, steps, elapsed = observations
    accepted = defaultdict(list)
    for (doc, name), indices, margin in chosen:
        if margin > auto.CUTOFF:
            accepted[doc, identities[doc].get(name, name)].append(set(indices))
    confirmed = query["positive"] & accepted.keys()
    support = 0
    covered = 0
    work = query.get("original", query["work"])
    for doc, name in confirmed:
        required = gold[doc, work, name, query["role"]]
        support += any(bool(required) and required <= positions for positions in accepted[doc, name])
        covered += any(bool(required & positions) for positions in accepted[doc, name])
    return dict(group=query["group"], positives=len(confirmed),
                rivals=len(query["rivals"] & accepted.keys()),
                unjudged=len(accepted.keys() - query["positive"] - query["rivals"]),
                all_annotated=support, any_annotated=covered, pairs=len(chosen), forwards=steps,
                depth=phases.count("depth"), identity=phases.count("identity"), source=phases.count("source"),
                ms=elapsed)


def summarize(panels, reference="staged_depth"):
    aggregate = {}
    for style, results in panels.items():
        baseline = results[reference]
        groups = [row["group"] for row in baseline]
        aggregate[style] = {}
        for arm, items in results.items():
            if [row["pairs"] for row in items] != [row["pairs"] for row in baseline]:
                raise ValueError("Mismatched pair work")
            totals = {name: sum(row[name] for row in items) for name in items[0] if name not in ("group", "ms")}
            deltas = [(row["all_annotated"] - ref["all_annotated"]) / index.TOP
                      for row, ref in zip(items, baseline, strict=True)]
            aggregate[style][arm] = totals | dict(
                queries=len(items), p95_ms=float(np.quantile([row["ms"] for row in items], .95)),
                **{f"grounded_p5_change_vs_{reference.removesuffix('_depth')}":
                   clustered_paired_interval(deltas, groups)},
            )
    return aggregate


def run(snapshot, extraction, checkpoint, output):
    import torch

    if output.exists() or "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    documents = auto.load_documents(snapshot)
    raw = json.loads(extraction.read_bytes())
    if raw["inputs"] != index.HASHES or len(raw["annotations"]) != len(documents):
        raise ValueError("Changed NER inputs")
    spans = raw["annotations"]
    _, annotated = index.mentions_index(documents)
    aliases, profiles = auto.auto_index(documents, spans, annotated)
    lookup = routing.AliasLookup(aliases)
    postings = index.bm25_index(documents)
    identities = auto.person_alias_judgments(documents)
    gold = windows.annotated_sentences(documents)
    base = index.study_queries(documents, 500)
    variants = routing.query_styles(base, documents)
    views = [dict(query, style="exact") for query in base] + [
        query for query in variants if query["style"] in ("transposition", "token_drop", "observed_alias")
    ]
    context = []
    for doc in documents:
        text = [" ".join(sentence) for sentence in doc["sents"]]
        context.append((text, evidence.profile(text)))
    tokenizer, model, info = oracle.load_reader(checkpoint)
    reader = oracle.reader_scorer(tokenizer, model)
    args = lookup, postings, profiles, documents, spans, reader, context
    for arm in ARMS:
        run_arm(views[0], arm, *args)
    result = defaultdict(lambda: defaultdict(list))
    for i, query in enumerate(views):
        shift = i % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            outcome = run_arm(query, arm, *args)
            result[query["style"]][arm].append(assess(query, outcome, gold, identities))
        if (i + 1) % 100 == 0:
            print(f"Completed {i+1} public role queries", flush=True)
    index.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=info["revision"],
        code_sha256=index.digest(Path(__file__)), projection_sha256=index.digest(extraction), inputs=index.HASHES,
        panels=summarize(result),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "extraction", "checkpoint", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
