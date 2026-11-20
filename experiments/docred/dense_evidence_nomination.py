"""Evaluate role-conditioned shortlist and evidence nomination at fixed verifier depth."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposals
from experiments.docred import nomination_evidence_windows as windows
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence
from experiments.docred import work_owner_contrast as upstream
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("automatic", "lexical_reserve", "dense_reserve", "dense_anchor")
NER_SHA = "76de6cb50fc5ea1b17a26bb9007893b4283b87b71fd57de0d09dbc74f013de1e"


def build_updated_index(revised):
    relationships = {}
    for doc_id, doc in enumerate(revised):
        names = [index.words(index.canonical(entity)) for entity in doc["vertexSet"]]
        for work, person, role in upstream.roles(doc):
            key = names[work], role
            relationships.setdefault(key, set()).add((doc_id, names[person]))
    return relationships


class Search:
    def __init__(self, documents, projection, encoder, reader):
        self.documents = documents
        self.projection = projection
        _, annotated = index.mentions_index(documents)
        aliases, self.profiles = auto.auto_index(documents, projection, annotated)
        self.lookup = routing.AliasLookup(aliases)
        self.postings = index.bm25_index(documents)
        self.encoder = encoder
        self.reader = reader
        texts = [tuple(" ".join(sentence) for sentence in doc["sents"]) for doc in documents]
        self.context = [(doc, evidence.profile(doc)) for doc in texts]
        sentences = list(dict.fromkeys(s for doc in texts for s in doc))
        ids = {s: i for i, s in enumerate(sentences)}
        self.offsets = [np.asarray([ids[s] for s in doc]) for doc in texts]
        self.unique_sentences = len(sentences)
        started = perf_counter()
        self.index = encoder.encode(sentences)
        self.index_seconds = perf_counter() - started

    def reserve(self, query, sources, selected, work, verb, method):
        used = set(selected)
        broad = {}
        if method in ("dense_reserve", "dense_anchor"):
            vector = self.encoder.encode([f"Find the source sentence that names who served as {verb} for {work}."],
                                          query=True)[0]
            ranks = {doc: self.index[self.offsets[doc]] @ vector for doc in sources}
            best = None
        for rank, doc in enumerate(sources):
            profile = self.profiles[doc]
            pool = proposals.pool(self.documents[doc], query, profile, self.projection[doc])
            if method == "lexical_reserve":
                names = [(name, tuple(sorted(spans))) for name, spans, _ in pool if (doc, name) not in used]
                broad[doc] = (names, *profile[1:])
                continue
            for name, mentions, _ in pool:
                if (doc, name) in used or not mentions:
                    continue
                anchor = max({i for i, _, _ in mentions}, key=lambda i: (float(ranks[doc][i]), -i))
                candidate = (float(ranks[doc][anchor]), -rank, name, doc, anchor)
                if best is None or candidate > best:
                    best = candidate
        if method == "lexical_reserve":
            candidates, _ = index.rank_people(query, sources, broad)
            return (candidates[0], None) if candidates else (None, None)
        if best is None:
            return None, None
        _, _, name, doc, anchor = best
        return (doc, name), anchor

    def run(self, query, method):
        started = perf_counter()
        sources = proposals.noisy_sources(query, self.postings, self.lookup)
        selected, _ = index.rank_people(query, sources, self.profiles)
        work, verb = " ".join(query["work"]), evidence.ROLES[query["role"]]
        anchor = None
        replaced = False
        if selected and method != "automatic":
            candidate, anchor = self.reserve(query, sources, selected, work, verb, method)
            if candidate is not None:
                selected[-1] = candidate
                replaced = True
        prompts, chosen = [], []
        for i, (doc, name) in enumerate(selected):
            person = " ".join(name)
            sentences, parsed = self.context[doc]
            window = evidence.choose(dict(work=work, role=query["role"]), person, parsed, "lexical_two")
            if method == "dense_anchor" and anchor is not None and i == len(selected) - 1 and len(window) == 2:
                # Keep an existing work anchor when inserting the dense mention.
                words, _, idf = parsed
                if anchor not in window:
                    other = max(window, key=lambda j: (evidence.contains(words[j], query["work"]),
                                                        evidence.coverage(query["work"], words[j], idf), -j))
                    window = tuple(sorted((anchor, other)))
            chosen.append(window)
            context = "\n".join(sentences[j] for j in window)
            prompts.append((f"Was {person} credited in the role of {verb} for {work}?", context))
        scores, clipped = self.reader(prompts) if prompts else ([], 0)
        elapsed = (perf_counter() - started) * 1000
        if clipped or not np.isfinite(scores).all():
            raise ValueError("Invalid verifier scores")
        return selected, chosen, [score > auto.CUTOFF for score in scores], elapsed, replaced


def evaluate(query, answer, identities, relationships, spans):
    shortlist, excerpts, accepted, ms, replaced = answer
    work = query.get("original", query["work"])
    role = query["role"]
    gold = relationships[work, role]
    observed = Counter(pairs=len(shortlist), sentences=sum(map(len, excerpts)), replaced=int(replaced))
    nominated, returned, grounded = set(), set(), set()
    for (doc, name), window, keep in zip(shortlist, excerpts, accepted, strict=True):
        pair = doc, identities[doc].get(name, name)
        nominated.add(pair)
        if not keep:
            continue
        returned.add(pair)
        label = spans[doc, work, pair[1], role]
        if pair in gold and label and label <= set(window):
            grounded.add(pair)
    observed.update(nominated_known=len(nominated & gold), accepted_known=len(returned & gold),
                    accepted_with_all_evidence=len(grounded), accepted_unjudged=len(returned-gold),
                    promoted_by_revision=len((returned & gold) - query["positive"]))
    return observed, ms


def run(snapshot, extended, projection, embedding, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    documents = auto.load_documents(snapshot)
    revised = [new for split in ("dev", "test") for _, new in upstream.documents(snapshot, extended, split)]
    if len(revised) != len(documents) or any(a["title"] != b["title"] for a,b in zip(documents, revised, strict=True)):
        raise ValueError("Different source order")
    if digest(projection) != NER_SHA:
        raise ValueError("Changed automatic projection")
    annotations = json.loads(projection.read_bytes())["annotations"]
    relationships = build_updated_index(revised)
    spans = windows.annotated_sentences(revised)
    # Preserve any original annotated evidence sets when the revision adds links
    # without adding sentence-level supervision.
    for key, original_locations in windows.annotated_sentences(documents).items():
        if not spans[key]:
            spans[key].update(original_locations)
    identities = auto.person_alias_judgments(documents)
    queries = auto.prepare_views(documents)
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    tokenizer, model, reader_info = oracle.load_reader(checkpoint)
    search = Search(documents, annotations, encoder, oracle.reader_scorer(tokenizer, model))
    for method in ARMS:
        search.run(queries[0], method)
    panels = {style: {method: [] for method in ARMS} for style in ("exact", "transposition")}
    for i, query in enumerate(queries):
        offset = i % len(ARMS)
        for method in ARMS[offset:] + ARMS[:offset]:
            result = search.run(query, method)
            counts, ms = evaluate(query, result, identities, relationships, spans)
            panels[query["style"]][method].append((counts, ms, query["group"]))
        if (i+1) % 60 == 0:
            print(f"Completed {i+1}/{len(queries)} queries", flush=True)
    summaries = {}
    for style, methods in panels.items():
        summaries[style] = {}
        baseline = methods["automatic"]
        for method, observations in methods.items():
            totals = sum((counts for counts, _, _ in observations), Counter())
            if any(a[0]["pairs"] != b[0]["pairs"] or a[0]["sentences"] != b[0]["sentences"]
                   for a, b in zip(observations, baseline, strict=True)):
                raise ValueError("Verifier slots or sentence budget differ")
            differences = [(a[0]["accepted_with_all_evidence"]-b[0]["accepted_with_all_evidence"])/index.TOP
                           for a, b in zip(observations, baseline, strict=True)]
            summaries[style][method] = dict(totals) | dict(
                queries=len(observations), p95_ms=float(np.quantile([ms for _, ms, _ in observations], .95)),
                annotated_grounded_p5_delta=clustered_paired_interval(differences,
                                                                     [group for _, _, group in observations]),
            )
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), projection=NER_SHA,
                            embedding=provenance["revision"], reader=reader_info["revision"],
                            docs=len(documents), unique_sentences=search.unique_sentences,
                            indexing_seconds=search.index_seconds, panels=summaries))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "extended", "projection", "embedding", "checkpoint", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
