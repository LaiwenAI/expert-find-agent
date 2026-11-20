"""Test person-marked evidence vectors for selective fixed-budget nomination."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import dense_evidence_nomination as baseline
from experiments.docred import role_local_person_rescue as rescue
from expert_search.qwen_embedding_torch import QwenTorchEmbedding


auto, index = baseline.auto, baseline.index
proposals, routing = baseline.proposals, baseline.routing
ARMS = ("automatic", "sentence_margin", "owner_margin", "sentence_calibrated", "owner_calibrated")
CAP = 0.30


def encode_mentions(doc, positions):
    for sent, start, end in sorted(positions):
        words = doc["sents"][sent]
        marked = " ".join([*words[:start], "[FOCAL PERSON]", *words[end:]])
        yield (sent, start, end), marked


class Search(baseline.Search):
    def __init__(self, documents, projection, encoder, reader):
        super().__init__(documents, projection, encoder, reader)
        views = {}
        for doc_id, (doc, profile, spans) in enumerate(zip(documents, self.profiles, projection, strict=True)):
            positions = {pos for _, mentions in profile[0] for pos in mentions}
            positions.update(pos for _, mentions in rescue.boundary_mentions(doc, spans, (), set())
                             for pos in mentions)
            positions.update((sent, start, end) for sent, words in enumerate(doc["sents"])
                             for _, start, end in rescue.plausible_names(words))
            views.update(((doc_id, *pos), text) for pos, text in encode_mentions(doc, positions))
        texts = list(dict.fromkeys(views.values()))
        ids = {text: i for i, text in enumerate(texts)}
        self.mention_ids = {span: ids[text] for span, text in views.items()}
        started = perf_counter()
        self.mentions = encoder.encode(texts)
        self.owner_index_seconds = perf_counter() - started
        self.cutoffs = {}

    def reserve(self, query, sources, selected, work, verb, method):
        kind, policy = method.split("_")
        cutoff = self.cutoffs[kind] if policy == "calibrated" else (
            -float("inf") if policy == "trial" else 0.)
        if cutoff is None:
            return None, None
        vector = self.encoder.encode([f"Find the evidence naming the focal person as {verb} for {work}."],
                                     query=True)[0]
        scores = {}
        for rank, doc in enumerate(sources):
            pool = proposals.pool(self.documents[doc], query, self.profiles[doc], self.projection[doc])
            for name, mentions, _ in pool:
                positions = sorted(mentions)
                if not positions:
                    continue
                if kind == "owner":
                    ids = [self.mention_ids[(doc, *position)] for position in positions]
                    affinities = self.mentions[ids] @ vector
                else:
                    ids = [self.offsets[doc][sent] for sent, _, _ in positions]
                    affinities = self.index[ids] @ vector
                scores[doc, name] = float(affinities.max()), -rank

        incumbent = selected[-1]
        choices = scores.keys() - set(selected)
        if not choices or incumbent not in scores:
            return None, None
        challenger = max(choices, key=lambda pair: (*scores[pair], pair))
        self.last_margin = scores[challenger][0] - scores[incumbent][0]
        return (challenger, None) if self.last_margin > cutoff else (None, None)

    def run(self, query, method):
        self.last_margin = None
        return super().run(query, method)


def calibrate(trials, reference, kind):
    options = [(margin, row, old) for (margin, row), old in zip(trials[kind], reference, strict=True)
               if margin is not None and margin > 0.]
    best_key = (0, 0, 0, 0)
    report = dict(cutoff=None, replacements=0, grounded_gain=0, known_gain=0, unjudged_delta=0)
    for cutoff in sorted({0., *(margin for margin, _, _ in options)}):
        picked = [(row, old) for margin, row, old in options if margin > cutoff]
        if not picked or len(picked) > int(CAP * len(reference)):
            continue
        def gain(field):
            return sum(row[field] - old[field] for row, old in picked)
        grounded = gain("accepted_with_all_evidence")
        known = gain("accepted_known")
        unknown = gain("accepted_unjudged")
        key = grounded, known, -unknown, -len(picked)
        if grounded > 0 and unknown <= 0 and key > best_key:
            best_key = key
            report = dict(cutoff=cutoff, replacements=len(picked), grounded_gain=grounded,
                          known_gain=known, unjudged_delta=unknown)
    return report | dict(queries=len(reference), eligible_positive_margins=len(options),
                         max_replacements=int(CAP * len(reference)))


def run(snapshot, extended, projection, embedding, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    documents = auto.load_documents(snapshot)
    revised = [new for split in ("dev", "test") for _, new in baseline.upstream.documents(snapshot, extended, split)]
    if len(revised) != len(documents) or any(a["title"] != b["title"] for a, b in zip(documents, revised, strict=True)):
        raise ValueError("Different source order")
    if baseline.digest(projection) != baseline.NER_SHA:
        raise ValueError("Changed automatic projection")
    annotations = json.loads(projection.read_bytes())["annotations"]
    relationships = baseline.build_updated_index(revised)
    spans = baseline.windows.annotated_sentences(revised)
    for key, original in baseline.windows.annotated_sentences(documents).items():
        if not spans[key]:
            spans[key].update(original)
    identities = auto.person_alias_judgments(documents)
    test = auto.prepare_views(documents)
    work_names = {query.get("original", query["work"]) for query in test}
    development = [q for q in index.study_queries(documents[:500], 0) if q["work"] not in work_names]
    provenance = baseline.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    tokenizer, model, reader_info = baseline.oracle.load_reader(checkpoint)
    search = Search(documents, annotations, encoder, baseline.oracle.reader_scorer(tokenizer, model))
    full_index = search.postings, search.lookup
    search.postings = index.bm25_index(documents[:500])
    search.lookup = routing.AliasLookup({key: {doc for doc in docs if doc < 500}
                                         for key, docs in full_index[1].aliases.items()
                                         if any(doc < 500 for doc in docs)})
    trials = {kind: [] for kind in ("owner", "sentence")}
    reference = []
    for query in development:
        answer = search.run(query, "automatic")
        counts, _ = baseline.evaluate(query, answer, identities, relationships, spans)
        reference.append(counts)
        for kind in trials:
            alternative = search.run(query, kind + "_trial")
            counts, _ = baseline.evaluate(query, alternative, identities, relationships, spans)
            trials[kind].append((search.last_margin, counts))
    gates = {kind: calibrate(trials, reference, kind) for kind in trials}
    search.cutoffs = {kind: gate["cutoff"] for kind, gate in gates.items()}
    search.postings, search.lookup = full_index
    print(f"Calibration complete: {gates}", flush=True)
    results = {style: {arm: [] for arm in ARMS} for style in ("exact", "transposition")}
    for index_, query in enumerate(test):
        order = index_ % len(ARMS)
        for arm in ARMS[order:] + ARMS[:order]:
            answer = search.run(query, arm)
            counts, ms = baseline.evaluate(query, answer, identities, relationships, spans)
            results[query["style"]][arm].append((counts, ms, query["group"]))
        if (index_ + 1) % 60 == 0:
            print(f"Completed {index_+1}/{len(test)} queries", flush=True)
    panels = {}
    for style, arms in results.items():
        panels[style] = {}
        control = arms["automatic"]
        for arm, rows in arms.items():
            totals = sum((counts for counts, _, _ in rows), Counter())
            if any(a[0]["pairs"] != b[0]["pairs"] or a[0]["sentences"] != b[0]["sentences"]
                   for a, b in zip(rows, control, strict=True)):
                raise ValueError("Expected fixed verifier and sentence budgets")
            differences = [(a[0]["accepted_with_all_evidence"] - b[0]["accepted_with_all_evidence"]) / index.TOP
                           for a, b in zip(rows, control, strict=True)]
            panels[style][arm] = dict(totals) | dict(
                queries=len(rows), p95_ms=float(np.quantile([ms for _, ms, _ in rows], .95)),
                annotated_grounded_p5_delta=clustered_paired_interval(differences, [group for _, _, group in rows]),
            )
    baseline.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=baseline.digest(Path(__file__)), embedding=provenance["revision"],
        reader=reader_info["revision"], docs=len(documents), sentence_encoding_seconds=search.index_seconds,
        owner_encoding_seconds=search.owner_index_seconds, owner_vectors=len(search.mentions),
        gates=gates, panels=panels,
    ))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "extended", "projection", "embedding", "checkpoint", "output"):
        args.add_argument(name, type=Path)
    run(**vars(args.parse_args()))
