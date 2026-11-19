"""Test fixed-size evidence windows guided by the candidate nomination score."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import contextual_role_nomination as nomination
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as prior
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence

WINDOWS = ("lexical", "anchored")


def lexical_window(case, person, parsed):
    return evidence.choose(case, person, parsed, "lexical_two")


def anchored_window(query, docs, nominee, documents, profiles, spans, mode, model, parsed, lexical):
    if len(lexical) < 2:
        return lexical
    doc, person = nominee
    document = documents[doc]
    profile = profiles[doc]
    rank = docs.index(doc)
    mentions, origins = next((positions, flags) for name, positions, flags in
                             prior.pool(document, query, profile, spans[doc]) if name == person)
    positions = sorted(mentions)
    cue_words = set(index.words(index.ROLES[query["role"]]))
    vectors = []
    for position in positions:
        singleton = (position,)
        vector = prior.candidate_features(document, query, profile, person, singleton, origins, rank)
        if mode == "context":
            local = nomination.mention_context(document["sents"], singleton, cue_words)
            vector = np.concatenate((vector, local))
        vectors.append(vector)

    # The best mention is a local probe of the nominated candidate's score.
    # Its vector need not explain the full pooled score.
    anchor = positions[int(np.argmax(model(np.stack(vectors))))][0]
    if anchor in lexical:
        return lexical
    words, _, idf = parsed
    work = query["work"]
    companion = max(lexical, key=lambda i: (
        evidence.contains(words[i], work), evidence.coverage(work, words[i], idf), -i,
    ))
    return tuple(sorted((anchor, companion)))


def assess_windows(query, mode, policy, get_docs, get_baseline, get_features, models, gates,
                   documents, profiles, spans, context, reader):
    start = perf_counter()
    docs = get_docs(query)
    candidates = get_baseline(query, docs)
    nominee, margin = nomination.proposal(query, docs, mode, candidates, get_features, models[mode])
    swapped = margin > gates[mode]["threshold"]
    if swapped:
        candidates = [*candidates[:-1], nominee]

    work = " ".join(query["work"])
    role = evidence.ROLES[query["role"]]
    case = dict(work=work, role=query["role"])
    prompts, windows = [], []
    for i, (doc, name) in enumerate(candidates):
        text, parsed = context[doc]
        person = " ".join(name)
        selected = lexical_window(case, person, parsed)
        if swapped and policy == "anchored" and i == len(candidates) - 1:
            selected = anchored_window(query, docs, nominee, documents, profiles, spans,
                                       mode, models[mode], parsed, selected)
        assert len(selected) == min(2, len(text))
        windows.append(selected)
        prompts.append((f"Was {person} credited in the role of {role} for {work}?",
                        "\n".join(text[sentence] for sentence in selected)))

    margins, clipped = reader(prompts) if prompts else ([], 0)
    elapsed = (perf_counter() - start) * 1000
    if clipped or not np.isfinite(margins).all():
        raise ValueError("Invalid reader result")
    return candidates, windows, margins, swapped, elapsed


def annotated_sentences(documents):
    labels = defaultdict(set)
    for i, doc in enumerate(documents):
        entities = doc["vertexSet"]
        names = [index.words(index.canonical(entity)) for entity in entities]
        for label in doc["labels"]:
            head, tail, role = (label[field] for field in ("h", "t", "r"))
            if role in index.ROLES and entities[tail][0]["type"] == "PER":
                labels[i, names[head], names[tail], role].update(label.get("evidence", ()))
    return labels


def observe(views, resolve, gold, args):
    report = defaultdict(lambda: defaultdict(list))
    arms = tuple((mode, policy) for mode in nomination.MODES for policy in WINDOWS)
    for i, query in enumerate(views):
        offset = i % len(arms)
        results = {(mode, policy): assess_windows(query, mode, policy, *args)
                   for mode, policy in arms[offset:] + arms[:offset]}

        for mode, policy in arms:
            baseline, base_windows, base_margins, _, _ = results[mode, "lexical"]
            candidates, windows, margins, swapped, elapsed = results[mode, policy]
            if candidates != baseline:
                raise ValueError("Evidence window changed nomination")
            identities = list(map(resolve, candidates))
            positive, rival = query["positive"], query["rivals"]
            accepted = {pair for pair, score in zip(identities, margins, strict=True) if score > auto.CUTOFF}
            prior_accepted = {pair for pair, score in zip(identities, base_margins, strict=True)
                              if score > auto.CUTOFF}
            last_windows = bool(windows and windows[-1] != base_windows[-1])
            earlier = sum((a > auto.CUTOFF) != (b > auto.CUTOFF)
                          for a, b in zip(margins[:-1], base_margins[:-1], strict=True))
            gold_window = set()
            if swapped and identities[-1] in positive:
                doc, person = identities[-1]
                work = query.get("original", query["work"])
                gold_window = gold[doc, work, person, query["role"]]
            found, lost = (accepted - prior_accepted) & positive, (prior_accepted - accepted) & positive
            report[query["style"]][mode, policy].append(dict(
                group=query["group"], slots=len(candidates), sentences=sum(map(len, windows)),
                swaps=int(swapped), changed_last_window=int(last_windows), changed_earlier_acceptances=earlier,
                positive=len(accepted & positive), rival=len(accepted & rival), unjudged=len(accepted - positive - rival),
                rescued_positive=len(found), lost_positive=len(lost), annotated_last=int(bool(gold_window)),
                any_annotated_sentence=int(bool(gold_window and gold_window.intersection(windows[-1]))),
                all_annotated_sentences=int(bool(gold_window and gold_window <= set(windows[-1]))), ms=elapsed,
            ))
    return report


def summarize(observations):
    panels = {}
    for style, arms in observations.items():
        panels[style] = {}
        for mode in nomination.MODES:
            baseline = arms[mode, "lexical"]
            groups = [row["group"] for row in baseline]
            panels[style][mode] = {}
            for policy in WINDOWS:
                rows = arms[mode, policy]
                assert groups == [row["group"] for row in rows]
                sums = {field: sum(row[field] for row in rows) for field in rows[0] if field not in ("ms", "group")}
                gains = [(row["positive"] - base["positive"]) / index.TOP
                         for row, base in zip(rows, baseline, strict=True)]
                panels[style][mode][policy] = sums | dict(
                    queries=len(rows), p95_ms=float(np.quantile([row["ms"] for row in rows], .95)),
                    positive_p5_delta=clustered_paired_interval(gains, groups),
                )
    return panels


def run(snapshot, projection, checkpoint, output):
    import os
    import torch

    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ or output.exists():
        raise ValueError("Require Slurm CUDA and a new output file")
    documents = auto.load_documents(snapshot)
    extracted = json.loads(projection.read_bytes())
    if extracted["inputs"] != index.HASHES or len(extracted["annotations"]) != len(documents):
        raise ValueError("Incorrect NER projection")
    spans = extracted["annotations"]
    _, annotated = index.mentions_index(documents)
    aliases, profiles = auto.auto_index(documents, spans, annotated)
    lookup = routing.AliasLookup(aliases)
    postings = index.bm25_index(documents)
    identities = auto.person_alias_judgments(documents)
    test = auto.prepare_views(documents)
    held_names = {query.get("original", query["work"]) for query in test}
    development = documents[:500]
    dev = [query for query in index.study_queries(development, 0) if query["work"] not in held_names]
    split = sorted({query["group"] for query in dev}, key=lambda group: routing.stable("context-v1", group))
    validation_groups = set(split[:max(1, len(split) // 4)])
    document_groups = {"validate": validation_groups, "train": set(range(500)) - validation_groups}

    def get_docs(query):
        return prior.noisy_sources(query, postings, lookup)

    def get_baseline(query, docs):
        return index.rank_people(query, docs, profiles)[0]

    def get_features(query, docs, mode):
        return nomination.features(query, docs, documents, profiles, spans, mode == "context")

    def resolve(pair):
        doc, name = pair
        return doc, identities[doc].get(name, name)

    tokenizer, reader, info = oracle.load_reader(checkpoint)
    score = oracle.reader_scorer(tokenizer, reader)
    models, training = nomination.train(nomination.MODES, dev, validation_groups, get_docs, get_features,
                                        resolve, document_groups)
    validation = [query for query in dev if query["group"] in validation_groups]
    validation += [query for query in routing.query_styles(validation, development)
                   if query["style"] == "transposition"]
    gates = {mode: nomination.calibrate(validation, mode, models[mode], get_docs, get_baseline,
                                        get_features, resolve) for mode in nomination.MODES}

    context = []
    for doc in documents:
        lines = [" ".join(row) for row in doc["sents"]]
        context.append((lines, evidence.profile(lines)))
    gold = annotated_sentences(documents)
    args = (get_docs, get_baseline, get_features, models, gates, documents, profiles, spans, context, score)
    observe(test[:6], resolve, gold, args)
    panels = summarize(observe(test, resolve, gold, args))
    index.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=info["revision"],
        code_sha256=index.digest(Path(__file__)), nominator_sha256=index.digest(Path(nomination.__file__)),
        projection_sha256=index.digest(projection), inputs=index.HASHES, training=training, gates=gates, panels=panels,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "projection", "checkpoint", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
