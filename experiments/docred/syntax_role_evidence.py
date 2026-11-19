"""Use offline grammatical role links to select candidates and evidence in one batch."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import dispatch_recovery_actions as dispatch
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import nomination_evidence_windows as windows
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence


ARMS = ("single_batch", "proximity", "syntax_candidate", "syntax_window", "syntax_joint")
ROLE_LEMMAS = {
    "P50": {"author", "write"}, "P57": {"director", "direct"},
    "P58": {"screenwriter", "script"}, "P86": {"composer", "compose"},
    "P112": {"founder", "found", "establish"}, "P161": {"star", "play", "portray"},
    "P162": {"producer", "produce"}, "P170": {"creator", "create", "design"},
    "P175": {"performer", "perform", "sing", "singer"}, "P178": {"developer", "develop"},
    "P676": {"lyricist"},
}


def credit_links(sentence):
    """Return role predicates with explicit credited-person token positions."""
    links = []
    for predicate in sentence:
        if predicate.pos_ not in ("NOUN", "VERB"):
            continue
        lemma = predicate.lemma_.lower()
        roles = [role for role, terms in ROLE_LEMMAS.items() if lemma in terms]
        if not roles:
            continue
        parent = predicate
        if parent.dep_ == "conj" and parent.head.pos_ == predicate.pos_:
            parent = parent.head
        if any(child.dep_ == "neg" for head in {predicate, parent} for child in head.children):
            continue
        local_subjects = [tok for tok in predicate.children if tok.dep_.startswith("nsubj")]
        active = [tok for tok in predicate.children if tok.dep_ == "nsubj"]
        if not local_subjects and predicate.dep_ == "conj":
            active = [tok for tok in parent.children if tok.dep_ == "nsubj"]
        agents = [obj for tok in predicate.children if tok.dep_ == "agent"
                  for obj in tok.children if obj.dep_ == "pobj"]
        if not agents and parent != predicate and not local_subjects:
            agents = [obj for tok in parent.children if tok.dep_ == "agent"
                      for obj in tok.children if obj.dep_ == "pobj"]
        nominal = [predicate.head] if predicate.dep_ == "appos" else []
        if predicate.dep_ == "attr" and predicate.head.lemma_.lower() == "be":
            nominal.extend(tok for tok in predicate.head.children if tok.dep_ == "nsubj")
        for actor in set(active + agents + nominal):
            for member in (actor, *actor.conjuncts):
                links.extend((role, member.i) for role in roles)
    return sorted(set(links))


def extract(documents):
    import spacy
    from spacy.tokens import Doc

    if spacy.__version__ != auto.ENGINE_VERSION:
        raise ValueError("Unexpected spaCy runtime")
    parser = spacy.load(auto.MODEL, enable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    if parser.meta["version"] != auto.MODEL_VERSION:
        raise ValueError("Unexpected parser checkpoint")
    spans = [[] for _ in documents]
    links = [defaultdict(set) for _ in documents]
    references = [(i, j, tokens) for i, doc in enumerate(documents) for j, tokens in enumerate(doc["sents"])]
    raw = (Doc(parser.vocab, words=tokens) for _, _, tokens in references)
    for (doc, sent, _), parsed in zip(references, parser.pipe(raw, batch_size=96), strict=True):
        spans[doc].extend((sent, ent.start, ent.end, ent.label_) for ent in parsed.ents
                          if ent.label_ == "PERSON" or ent.label_ in auto.WORK_LABELS)
        for role, actor in credit_links(parsed):
            links[doc][role].add((sent, actor))
    return spans, links


def role_sentences(role, mentions, links):
    positions = links.get(role, ())
    return {s for s, a, b in mentions if any(sent == s and a <= actor < b for sent, actor in positions)}


def candidate_details(query, sources, documents, profiles, spans, links, use_syntax):
    details = {}
    for rank, doc in enumerate(sources):
        people, works, role_positions = profiles[doc]
        work = works.get(query["work"], set())
        cues = [position for term in index.words(index.ROLES[query["role"]])
                for position in role_positions.get(term, ())]
        for name, mentions, _ in proposal.pool(documents[doc], query, profiles[doc], spans[doc]):
            if use_syntax:
                attached = role_sentences(query["role"], mentions, links[doc])
                local = any(any(abs(s - w) <= 1 for w in work) for s in attached)
                strength = 2 if local else int(bool(attached) and not work)
            else:
                distances = [min(abs(a - pos), abs(b - 1 - pos)) for s, a, b in mentions
                             for sent, pos in cues if sent == s and (not work or s in work)]
                strength = max(0, 5 - min(distances, default=5))
            details[doc, name] = strength, -rank, len(mentions)
    return details


def replace_tail(prior, details):
    if not prior:
        return prior
    anchor = prior[-1]
    current = details[anchor][0]
    eligible = [pair for pair in details.keys() - set(prior) if details[pair][0] > current]
    if not eligible:
        return prior
    chosen = min(eligible, key=lambda pair: tuple(-x for x in details[pair]) + (pair,))
    return prior[:-1] + [chosen]


def choose_window(query, person, parsed, attached):
    case = dict(work=" ".join(query["work"]), role=query["role"])
    selected = evidence.choose(case, " ".join(person), parsed, "lexical_two")
    if not attached or attached.intersection(selected):
        return selected
    words, _, idf = parsed
    work = tuple(evidence.tokens(case["work"]))
    def work_match(s):
        return evidence.coverage(work, words[s], idf), -s

    return tuple(sorted((max(attached, key=work_match), max(selected, key=work_match))))


def measure(query, arm, lookup, postings, profiles, documents, spans, links, read, context):
    began = perf_counter()
    sources, _, _ = routing.route(query, postings, lookup, "swap_then_guard")
    prior = index.rank_people(query, sources, profiles)[0]
    pairs = prior
    if arm in ("proximity", "syntax_candidate", "syntax_joint"):
        details = candidate_details(query, sources, documents, profiles, spans, links, arm != "proximity")
        pairs = replace_tail(prior, details)
    indices, prompts = [], []
    for doc, person in pairs:
        text, parsed, mentions = context[doc]
        attached = set()
        if arm in ("syntax_window", "syntax_joint"):
            person_spans = mentions.get(person)
            if person_spans is None:
                person_spans = next((positions for name, positions, _ in proposal.pool(
                    documents[doc], query, profiles[doc], spans[doc]) if name == person), ())
            attached = role_sentences(query["role"], person_spans, links[doc])
        chosen = choose_window(query, person, parsed, attached)
        indices.append(chosen)
        role = evidence.ROLES[query["role"]]
        prompts.append((f"Was {' '.join(person)} credited in the role of {role} for {' '.join(query['work'])}?",
                        "\n".join(text[i] for i in chosen)))
    margins, clipped = read(prompts) if prompts else ([], 0)
    if clipped or not np.isfinite(margins).all():
        raise ValueError("Invalid model output")
    return (list(zip(pairs, indices, margins, strict=True)), [], int(bool(pairs)),
            (perf_counter() - began) * 1000), pairs != prior


def run(snapshot, holdout, sha256, checkpoint, output):
    import torch

    if output.exists() or "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    if index.digest(holdout) != sha256:
        raise ValueError("Changed holdout snapshot")
    docs = auto.load_documents(snapshot)
    extra = json.loads(holdout.read_bytes())
    known_titles = {index.words(doc["title"]) for doc in docs}
    extra = [doc for doc in extra if index.words(doc["title"]) not in known_titles]
    documents = docs + extra
    started = perf_counter()
    spans, links = extract(documents)
    parse_seconds = perf_counter() - started
    _, annotated = index.mentions_index(documents)
    aliases, profiles = auto.auto_index(documents, spans, annotated)
    lookup = routing.AliasLookup(aliases)
    postings = index.bm25_index(documents)
    identities = auto.person_alias_judgments(documents)
    gold = windows.annotated_sentences(documents)
    development = {q["work"] for q in index.study_queries(docs, 0)}
    queries = [q for q in index.study_queries(documents, len(docs)) if q["work"] not in development]
    variants = routing.query_styles(queries, documents)
    views = [dict(q, style="exact") for q in queries] + [q for q in variants if q["style"] == "token_drop"]
    context = []
    for doc, (people, _, _) in zip(documents, profiles, strict=True):
        text = [" ".join(sentence) for sentence in doc["sents"]]
        context.append((text, evidence.profile(text), dict(people)))
    tokenizer, model, info = oracle.load_reader(checkpoint)
    read = oracle.reader_scorer(tokenizer, model)
    measured = defaultdict(lambda: defaultdict(list))
    for arm in ARMS:
        measure(views[0], arm, lookup, postings, profiles, documents, spans, links, read, context)
    for i, query in enumerate(views):
        shift = i % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            outcome, replaced = measure(query, arm, lookup, postings, profiles, documents, spans, links, read, context)
            measured[query["style"]][arm].append(
                dispatch.assess(query, outcome, gold, identities) | {"replaced": int(replaced)}
            )
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(views)} holdout queries", flush=True)
    results = dispatch.summarize(measured, reference="single_batch")
    index.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=info["revision"], inputs=index.HASHES, upstream_train_sha256=sha256,
        documents=len(documents), holdout_documents=len(extra), parse_seconds=parse_seconds,
        code_sha256=index.digest(Path(__file__)), panels=results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "holdout", "checkpoint", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("sha256")
    run(**vars(cli.parse_args()))
