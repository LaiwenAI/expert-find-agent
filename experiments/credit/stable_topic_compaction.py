"""Separate batch geometry from role-safe source compaction in topical verification."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_reranker import pack_tokens
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("base", "compact", "fixed_width", "margin_guard", "grouped", "grouped_skip")
GROUP_SIZE = 2


class BatchReader:
    def __init__(self, reranker):
        self.reader = reranker

    def encode(self, query, texts):
        encode = self.reader.tokenizer.encode
        rows = []
        for text in texts:
            body = encode(f"<Instruct>: {pilot.TASK}\n<Query>: {query}\n<Document>: {text}",
                          add_special_tokens=False)
            tokens, raw = pack_tokens(self.reader.prefix, body, self.reader.suffix, 1024)
            if raw > len(tokens):
                raise ValueError("Unexpected topical verifier truncation")
            rows.append(dict(input_ids=tokens))
        return rows

    def score(self, rows, indexes, width=None):
        import torch

        if not indexes:
            return [], 0
        data = [rows[i] for i in indexes]
        pad = {} if width is None else {"max_length": width}
        with torch.inference_mode():
            inputs = self.reader.tokenizer.pad(
                data, padding=True if width is None else "max_length", return_tensors="pt", **pad).to("cuda")
            logits = self.reader.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.reader.labels].float()
            result = (logits[:, 0] - logits[:, 1]).cpu().tolist()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite score")
        return result, inputs["input_ids"].numel()


def eligible(catalog, articles, pair, cover):
    postings = catalog.person[pair]
    known = catalog.visible[cover]
    return [i for i, article in enumerate(articles) if article in postings or article not in known]


def select_groups(total, active, skip):
    for start in range(0, total, GROUP_SIZE):
        group = list(range(start, min(total, start + GROUP_SIZE)))
        if not skip or any(i in active for i in group):
            yield group


def margin_risk(scores, tolerance):
    if any(abs(score - holdout.CUTOFF) <= tolerance for score in scores):
        return True
    positive = sorted(s for s in scores if s > holdout.CUTOFF)
    return any(b - a <= 2 * tolerance for a, b in zip(positive, positive[1:]))


def compute(reader, rows, active, arm, tolerance):
    full = list(range(len(rows)))
    if arm in ("grouped", "grouped_skip"):
        logits, slots = {}, 0
        for group in select_groups(len(rows), set(active), arm == "grouped_skip"):
            values, used = reader.score(rows, group)
            logits.update(zip(group, values, strict=True))
            slots += used
        return logits, slots, False
    selected = full if arm == "base" else active
    width = max(len(row["input_ids"]) for row in rows) if arm == "fixed_width" else None
    values, slots = reader.score(rows, selected, width=width)
    if arm == "margin_guard" and margin_risk(values, tolerance):
        values, extra = reader.score(rows, full)
        return dict(zip(full, values, strict=True)), slots + extra, True
    return dict(zip(selected, values, strict=True)), slots, False


def evidence(catalog, topic, pair, articles, values):
    people = {}
    for pos in sorted(values, key=lambda p: -values[p]):
        if values[pos] <= holdout.CUTOFF:
            continue
        for owner in catalog.bindings.get((articles[pos], pair), ()):
            people.setdefault(owner, articles[pos])
            if len(people) == pilot.TOP:
                break
        if len(people) == pilot.TOP:
            break
    grounded = sum(article in catalog.topics[topic] for article in people.values())
    return dict(issued=len(people), grounded=grounded,
                cited_off_label=len(people) - grounded, witnesses=tuple(people.items()))


def queries(catalog):
    topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= 4)
    return list(pilot.query_panel(catalog, topics))


def calibration(reader, catalog):
    changes = []
    requests = 0
    for topic, _, pair in queries(catalog):
        articles = catalog.rank(topic)[:8]
        inputs = reader.encode(f"Research about {topic}", [catalog.texts[i] for i in articles])
        canonical, _, _ = compute(reader, inputs, [], "base", 0.)
        for cover in pilot.CACHE:
            active = eligible(catalog, articles, pair, cover)
            current, _, _ = compute(reader, inputs, active, "compact", 0.)
            changes.extend(abs(canonical[i] - current[i]) for i in active)
            requests += 1
    return dict(tolerance=max(changes), requests=requests, retained_scores=len(changes))


def summarize(options, reference, compare_to):
    rows = [row["result"] for row in options]
    control = [row["result"] for row in reference]
    pairs = list(zip(options, reference, strict=True))
    result = holdout.analyze(rows, control)
    result.update(
        compares_to=compare_to,
        active_score_differences=sum(a["values"][i] != b["values"][i]
                                     for a, b in pairs for i in a["active"]),
        active_score_threshold_flips=sum((a["values"][i] > holdout.CUTOFF) != (b["values"][i] > holdout.CUTOFF)
                                         for a, b in pairs for i in a["active"]),
        max_active_score_difference=max((abs(a["values"][i]-b["values"][i])
                                        for a, b in pairs for i in a["active"]), default=0.),
        packed_token_slots=sum(row["slots"] for row in options),
        rechecks=sum(row["rechecked"] for row in options),
    )
    return result


def run(pilot_projection, projection, manifest, embedding, reranking, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if (pilot.digest(pilot_projection), pilot.digest(projection)) != (
            frozen["original_projection_sha256"], frozen["projection_sha256"]):
        raise ValueError("Unexpected topic projections")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reranker = Reranker(reranking)
    reader = BatchReader(reranker)
    development = pilot.Catalog(json.loads(pilot_projection.read_bytes())["sources"], encoder, reranker)
    target = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, reranker)
    gate = calibration(reader, development)
    test = queries(target)
    example, _, _ = test[0]
    example_docs = [target.texts[i] for i in target.rank(example)[:8]]
    rows = reader.encode(f"Research about {example}", example_docs)
    original, _ = reranker.score_pairs([(f"Research about {example}", t) for t in example_docs], task=pilot.TASK)
    assert reader.score(rows, list(range(len(rows))))[0] == original
    observations = {c: {a: [] for a in ARMS} for c in pilot.CACHE}
    for j, (topic, _, pair) in enumerate(test):
        started = perf_counter()
        articles = target.rank(topic)[:8]
        encoded = reader.encode(f"Research about {topic}", [target.texts[i] for i in articles])
        common_ms = (perf_counter() - started) * 1000
        shift = j % len(ARMS)
        for cover in pilot.CACHE:
            for arm in ARMS[shift:] + ARMS[:shift]:
                started = perf_counter()
                active = eligible(target, articles, pair, cover)
                values, slots, rechecked = compute(reader, encoded, active, arm, gate["tolerance"])
                results = evidence(target, topic, pair, articles, values)
                elapsed = common_ms + (perf_counter() - started) * 1000
                observations[cover][arm].append(dict(
                    active=active, values=values, slots=slots, rechecked=rechecked,
                    result=dict(results, topic=topic, ms=elapsed, reads=len(values),
                                characters=sum(len(target.texts[articles[i]]) for i in values)),
                ))
    report = {}
    for cover, arms in observations.items():
        report[cover] = {}
        for arm, rows in arms.items():
            baseline = "grouped" if arm == "grouped_skip" else "base"
            report[cover][arm] = summarize(rows, arms[baseline], baseline)
            if arm == "grouped_skip":
                report[cover][arm]["vs_eight"] = summarize(rows, arms["base"], "base")
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=pilot.digest(Path(__file__)), manifest_sha256=pilot.digest(manifest),
        embedding=encoder_info["revision"], reranker=reranker.provenance["revision"],
        calibration=gate, queries=len(test), topics=len({t for t, _, _ in test}), report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot_projection", "projection", "manifest", "embedding", "reranking", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
