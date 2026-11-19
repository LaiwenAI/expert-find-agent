"""Test name-masked role context for replacing one fixed verification slot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter
from zlib import crc32

import numpy as np

from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as prior
from experiments.docred import noisy_role_source_routing as routing

MODES = ("geometry", "context")
ARMS = ("automatic", *MODES)
BUCKETS = 512
WINDOW = 7
CAP = .30


def mention_context(sentences, spans, cue_words):
    vector = np.zeros(BUCKETS, dtype=np.float32)

    def term(token):
        if token.casefold() in cue_words:
            return "<role>"
        if token in (",", ".", ":", ";", "(", ")", "-", "–", "—"):
            return token
        if token and token[0].isupper():
            return "<name>"
        return token.casefold() if token.isalpha() else "<other>"

    for sent_index, start, end in spans:
        sentence = sentences[sent_index]
        left = tuple(map(term, sentence[max(0, start - WINDOW):start]))
        right = tuple(map(term, sentence[end:end + WINDOW]))
        for side, sequence in (("L", left[::-1]), ("R", right)):
            for length in (1, 2, 3):
                if len(sequence) >= length:
                    gram = " ".join(sequence[:length])
                    vector[crc32(f"{side}:adj:{gram}".encode()) % BUCKETS] = 1
            for position, token in enumerate(sequence):
                if token != "<role>":
                    continue
                bridge = " ".join(sequence[:position])
                vector[crc32(f"{side}:role:{bridge}".encode()) % BUCKETS] = 1
    return vector


def features(query, docs, documents, profiles, spans, use_context):
    if not use_context:
        return prior.candidates(query, docs, documents, profiles, spans)
    cue_words = set(index.words(index.ROLES[query["role"]]))
    result = {}
    for rank, doc in enumerate(docs):
        document = documents[doc]
        profile = profiles[doc]
        for name, mentions, origins in prior.pool(document, query, profile, spans[doc]):
            base = prior.candidate_features(document, query, profile, name, mentions, origins, rank)
            local = mention_context(document["sents"], mentions, cue_words)
            result[doc, name] = np.concatenate((base, local))
    return result


def train(modes, queries, validation_groups, get_docs, get_features, resolve, document_groups):
    import torch

    positives_by_mode = {}
    models = {}
    for mode in modes:
        contrasts = {"train": [], "validate": []}
        for query in queries:
            split = "validate" if query["group"] in validation_groups else "train"
            docs = [doc for doc in get_docs(query) if doc in document_groups[split]]
            vectors = get_features(query, docs, mode)
            for doc in docs:
                positive = [v for pair, v in vectors.items()
                            if pair[0] == doc and resolve(pair) in query["positive"]]
                rival = [v for pair, v in vectors.items()
                         if pair[0] == doc and resolve(pair) in query["rivals"]]
                contrasts[split].extend(good - bad for good in positive for bad in rival)
        if not all(contrasts.values()):
            raise ValueError("Insufficient source-disjoint role contrasts")
        training = torch.as_tensor(np.stack(contrasts["train"]), device="cuda")
        validation = torch.as_tensor(np.stack(contrasts["validate"]), device="cuda")
        weight = torch.zeros(training.shape[1], device="cuda", requires_grad=True)
        optimizer = torch.optim.AdamW([weight], lr=.025, weight_decay=.04)
        best = None
        started = perf_counter()
        for step in range(1, 181):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(training @ weight)).mean() + .002 * weight.square().sum()
            loss.backward()
            optimizer.step()
            if step % 15 == 0:
                with torch.no_grad():
                    margins = validation @ weight
                    value = float(torch.nn.functional.softplus(-margins).mean())
                    if best is None or value < best[0]:
                        best = value, step, float((margins > 0).float().mean()), weight.detach().clone()
        torch.cuda.synchronize()
        loss, step, ordering, selected = best
        weights = selected.cpu().numpy()
        models[mode] = lambda vectors, weights=weights: vectors @ weights
        positives_by_mode[mode] = dict(
            training_pairs=len(training), validation_pairs=len(validation), dimensions=len(weights),
            selected_step=step, validation_loss=loss, validation_pair_accuracy=ordering,
            gpu_fit_seconds=perf_counter() - started,
        )
    return models, positives_by_mode


def proposal(query, docs, mode, baseline, get_features, score):
    if not baseline:
        return None, 0.
    vectors = get_features(query, docs, mode)
    # Reuse the previous study's name-origin and local cue guard.
    return prior.rank_replacement(vectors, baseline[-1], baseline, score, guarded=True)


def calibrate(queries, mode, score, get_docs, get_baseline, get_features, resolve):
    observations = []
    for query in queries:
        docs = get_docs(query)
        baseline = get_baseline(query, docs)
        candidate, margin = proposal(query, docs, mode, baseline, get_features, score)
        if margin <= 0:
            continue
        before = set(map(resolve, baseline))
        after = set(map(resolve, [*baseline[:-1], candidate]))
        support, rivals = query["positive"], query["rivals"]
        observations.append((margin, len(after & support) - len(before & support),
                             len(after & rivals) - len(before & rivals), len(after - support - rivals) - len(before - support - rivals)))
    cap = int(CAP * len(queries))
    best = (0, 0, 0, 0, max((row[0] for row in observations), default=0.))
    for threshold in sorted({0., *(row[0] for row in observations)}):
        replacements = [row for row in observations if row[0] > threshold]
        gain, rivals, unknown = (sum(row[col] for row in replacements) for col in (1, 2, 3))
        if len(replacements) <= cap and rivals <= 0:
            best = max(best, (gain, -rivals, -unknown, -len(replacements), threshold))
    gain, rivals, unknown, count, threshold = best
    return dict(threshold=threshold, proposed_positive_delta=gain, proposed_rival_delta=-rivals,
                proposed_unjudged_delta=-unknown, replacements=-count, replacement_cap=cap,
                queries=len(queries))


def measure(views, get_docs, get_baseline, get_features, resolve, readers, gates, reader, context):
    result = defaultdict(lambda: defaultdict(list))
    for i, query in enumerate(views):
        offset = i % len(ARMS)
        for mode in ARMS[offset:] + ARMS[:offset]:
            start = perf_counter()
            docs = get_docs(query)
            baseline = get_baseline(query, docs)
            candidates = baseline
            if mode != "automatic":
                nominee, margin = proposal(query, docs, mode, baseline, get_features, readers[mode])
                if margin > gates[mode]["threshold"]:
                    candidates = [*baseline[:-1], nominee]
            decision = perf_counter()
            margins = reader(query, candidates, context)
            finish = perf_counter()
            identities = list(map(resolve, candidates))
            accepted = {pair for pair, margin in zip(identities, margins, strict=True) if margin > auto.CUTOFF}
            positive, rival = query["positive"], query["rivals"]
            result[query["style"]][mode].append(dict(
                group=query["group"], replaced=int(candidates != baseline), pairs=len(candidates),
                proposed_positive=len(set(identities) & positive), accepted_positive=len(accepted & positive),
                accepted_rival=len(accepted & rival), accepted_unjudged=len(accepted - positive - rival),
                decision_ms=(decision - start) * 1000, total_ms=(finish - start) * 1000,
            ))
    return result


def run(snapshot, projection, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
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
    dev_documents = documents[:500]
    dev = [query for query in index.study_queries(dev_documents, 0) if query["work"] not in held_names]
    split = sorted({query["group"] for query in dev}, key=lambda group: routing.stable("context-v1", group))
    validation_groups = set(split[:max(1, len(split) // 4)])
    document_groups = {"validate": validation_groups, "train": set(range(500)) - validation_groups}

    def get_docs(query):
        return prior.noisy_sources(query, postings, lookup)

    def get_baseline(query, docs):
        return index.rank_people(query, docs, profiles)[0]

    def get_features(query, docs, mode):
        return features(query, docs, documents, profiles, spans, mode == "context")

    def resolve(pair):
        doc, name = pair
        return doc, identities[doc].get(name, name)

    reader, preprocess, info = auto.reader_session(checkpoint)
    models, training = train(MODES, dev, validation_groups, get_docs, get_features, resolve, document_groups)
    validation = [query for query in dev if query["group"] in validation_groups]
    validation += [query for query in routing.query_styles(validation, dev_documents)
                   if query["style"] == "transposition"]
    gates = {mode: calibrate(validation, mode, models[mode], get_docs, get_baseline, get_features, resolve)
             for mode in MODES}

    context = []
    for doc in documents:
        text = [" ".join(row) for row in doc["sents"]]
        context.append((text, preprocess(text)))

    evaluate = (get_docs, get_baseline, get_features, resolve, models, gates, reader, context)
    measure(test[:6], *evaluate)
    panels = prior.aggregate(measure(test, *evaluate))
    index.write_json(output, dict(
        **info, code_sha256=index.digest(Path(__file__)), projection_sha256=index.digest(projection),
        inputs=index.HASHES, training=training, gates=gates, development_reference_docs=len(split),
        validation_reference_docs=len(validation_groups), panels=panels,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "projection", "checkpoint", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
