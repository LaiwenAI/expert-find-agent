"""Test exact publisher-proof filtering before live semantic reranking."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.requirement_packets import form_packet
from experiments.credit import elife_role_normalization_search as search
from experiments.credit import study_role_normalization as roles
from experiments.credit import study_role_normalization_batch as batch
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


ARMS = ("late_proof", "compact", "refill", "gap_refill")
LIMITS = (4, 8)
ROLE_PREFIX = "Role: "
PREFERENCE = "Optional: include a second reference."
OPEN_CONDITION = "The author must already hold written approval to release the raw data."


def postings(articles):
    """Index same-author intersections without melding roles across authors."""
    triples = defaultdict(set)
    native = frozenset(search.NATIVE.values())
    for i, article in enumerate(articles):
        for author in article["authors"]:
            for width in (2, 3):
                for fields in combinations(sorted(native.intersection(author["roles"])), width):
                    triples[fields].add(i)
    return triples


def panel(topics, index, complete):
    name_to_key = {name: key for key, name in search.NATIVE.items()}
    rows = []
    for width in (2, 3):
        for topic, tagged in sorted(topics.items()):
            tagged = tagged & complete
            if len(tagged) < 4:
                continue
            counts = {fields: len(sources & tagged)
                      for fields, sources in index.items() if len(fields) == width}
            possible = [fields for fields, count in counts.items() if 0 < count < len(tagged)]
            if not possible:
                continue
            # One rare and one common conjunction per topic, with deterministic ties.
            ordered = sorted(possible, key=lambda fields: (counts[fields], pilot.stable(topic, fields)))
            chosen = ordered[:1] + ordered[-1:] if len(ordered) > 1 else ordered
            for fields in chosen:
                seed = pilot.stable("proof-before-rerank-v1", topic, fields)
                keys = [name_to_key[name] for name in fields]
                phrased = [roles.PHRASES[key][seed[i] % len(roles.PHRASES[key])] for i, key in enumerate(keys)]
                text = "\n".join([*(f"{ROLE_PREFIX}{phrase}." for phrase in phrased), PREFERENCE])
                rows.append((topic, frozenset(fields), text, False))
                if width == 3:
                    rows.append((topic, frozenset(fields), f"{text}\n{OPEN_CONDITION}", True))
    return rows


def choose(ranked, complete, supported, arm, limit):
    initial = ranked[:limit]
    if arm == "late_proof":
        return initial
    scope = initial if arm in {"compact", "gap_refill"} else ranked
    return [source for source in scope
            if source in supported or source not in complete][:limit]


def bind(catalog, topic, fields, gold, selected, output):
    scores = catalog.verify(topic, selected)
    seen = {person for person, *_ in output}
    for source, score in sorted(zip(selected, scores, strict=True), key=lambda item: -item[1]):
        if score <= holdout.CUTOFF:
            continue
        article = catalog.articles[source]
        for author in article["authors"]:
            person = author["person"] or f'{article["doi"]}#{author["ordinal"]}'
            declared = set(author["roles"])
            if person in seen or not fields <= declared:
                continue
            seen.add(person)
            output.append((person, source, gold <= declared, source in catalog.topics[topic]))
            if len(output) == search.SEATS:
                return


def recommend(catalog, judge, row, complete, index, arm, limit):
    topic, gold, text, _ = row
    start = perf_counter()
    packet = form_packet(text, None)
    raw_clauses = [part.quote(text) for part in packet.unresolved]
    tasks = [part.removeprefix(ROLE_PREFIX).removesuffix(".") for part in raw_clauses
             if part.startswith(ROLE_PREFIX)]
    empty = dict(unresolved=1, reader_pairs=0, reader_calls=0, new_sources=0)
    if len(tasks) != len(raw_clauses) or not tasks:
        return (), empty, (perf_counter() - start) * 1000
    _, accepted, _ = batch.normalize(judge, tasks, "one_batch")
    if any(value not in search.NATIVE for value in accepted) or len(set(accepted)) != len(tasks):
        return (), empty, (perf_counter() - start) * 1000

    fields = frozenset(search.NATIVE[val] for val in accepted)
    ranked = catalog.rank(topic)
    supported = index.get(tuple(sorted(fields)), set())
    selected = choose(ranked, complete, supported, arm, limit)
    output = []
    bind(catalog, topic, fields, gold, selected, output)
    calls = int(bool(selected))
    if arm == "gap_refill" and len(output) < search.SEATS and len(selected) < limit:
        extra = choose(ranked, complete, supported, "refill", limit)[len(selected):]
        bind(catalog, topic, fields, gold, extra, output)
        selected.extend(extra)
        calls += int(bool(extra))
    return tuple(output), dict(
        unresolved=0, exact_field_mapping=int(fields == gold), reader_calls=calls,
        reader_pairs=len(selected), new_sources=len(set(selected) - set(ranked[:limit])),
    ), (perf_counter() - start) * 1000


def tagged_count(output):
    return sum(role and tag for _, _, role, tag in output)


def evaluate(catalog, judge, offset):
    complete = {i for i, article in enumerate(catalog.articles) if article["declaration_complete"]}
    index = postings(catalog.articles)
    rows = panel(catalog.topics, index, complete)
    settings = {"partial": complete & catalog.visible[50], "full": complete}
    if not rows or not settings["partial"]:
        raise ValueError("Empty coverage panel")
    runs = {}
    for mask, visible in settings.items():
        for limit in LIMITS:
            counts = defaultdict(lambda: defaultdict(Counter))
            times = defaultdict(lambda: defaultdict(list))
            deltas = defaultdict(lambda: defaultdict(list))
            groups = defaultdict(list)
            for arm in ARMS:
                recommend(catalog, judge, rows[0], visible, index, arm, limit)
            for i, row in enumerate(rows):
                topic, gold, _, open_condition = row
                cohort = "open_condition" if open_condition else f"width_{len(gold)}"
                groups[cohort].append(topic)
                shift = (i + offset) % len(ARMS)
                trials = {arm: recommend(catalog, judge, row, visible, index, arm, limit)
                          for arm in ARMS[shift:] + ARMS[:shift]}
                reference = trials["late_proof"][0]
                for arm, (found, counters, elapsed) in trials.items():
                    if open_condition and found:
                        raise AssertionError("Admitted a request with an unresolved mandatory condition")
                    matched = min(len(reference), len(found))
                    delta = tagged_count(found[:matched]) - tagged_count(reference[:matched])
                    counts[cohort][arm].update(
                        queries=1, issued=len(found), tag_backed=tagged_count(found),
                        without_requested_role=sum(not role for _, _, role, _ in found),
                        identical_output=int(found == reference), matched_issued=matched, matched_tag_delta=delta,
                        **counters,
                    )
                    deltas[cohort][arm].append(delta / search.SEATS)
                    times[cohort][arm].append(elapsed)
            runs[f"{mask}/{limit}"] = dict(complete_visible=len(visible), cohorts={
                cohort: {arm: dict(values, p95_ms=float(np.quantile(times[cohort][arm], .95)),
                                   matched_tag_p5_interval=holdout.interval(deltas[cohort][arm], groups[cohort]))
                         for arm, values in reports.items()}
                for cohort, reports in counts.items()
            })
    return runs


def main():
    import torch
    from experiments.followir.core17_verifier import Reranker

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("projection", "manifest", "reference", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    manifest = json.loads(args.manifest.read_bytes())
    reference = json.loads(args.reference.read_bytes())["run"]
    if digest(args.projection) != manifest["projection_sha256"]:
        raise ValueError("Unexpected eLife projection")
    embedding = Path("artifacts/models/qwen3-embedding-0.6b")
    enc_info = pilot.verify_checkpoint(embedding)
    enc = QwenTorchEmbedding(embedding)
    enc.model.config.use_cache = False
    reader = Reranker(Path("artifacts/models/qwen3-reranker-0.6b"))
    if (enc_info["revision"], reader.provenance["revision"]) != (reference["embedding"], reference["reranker"]):
        raise ValueError("Unexpected topical checkpoint")
    judge = Judge()
    roles.extended_ids(judge)
    articles = [dict(row, topics=row["keywords"]) for row in json.loads(args.projection.read_bytes())["sources"]]
    catalog = pilot.Catalog(articles, enc, reader)
    report = evaluate(catalog, judge, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
        projection_sha256=digest(args.projection), reference_sha256=digest(args.reference),
        decision_model_sha256=judge.sha, documents=len(articles), panels=report,
    ))


if __name__ == "__main__":
    main()
