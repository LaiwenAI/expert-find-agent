"""Select one answer window on development, then transfer to a later cohort."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.community.crossvalidated_witness import write
from experiments.community.evaluate_crossvalidated_witness import connected_groups
from experiments.community.matched_witness_rescue import TASK, checks, digest, evaluate, shortlist
from experiments.followir.constraint_window_verification import WINDOW_CHARS, select_window

POLICIES = ("lead", "targeted", "routed", "more_request", "explicit_prompt", "both_changes")
EXPLICIT = ("Verify that the answer text provides a method, explanation, or conclusion that directly helps "
            "answer the user's question. A matching topic or canonical question title alone is insufficient.")
REQUEST_CHARS = 950
EXTENDED_CHARS = 1800


def load(snapshot, manifest):
    receipt = json.loads(manifest.read_bytes())
    if receipt["snapshot_sha256"] != digest(snapshot):
        raise ValueError("Changed frozen snapshot")
    data = json.loads(snapshot.read_bytes())
    if sorted(map(int, data["documents"])) != receipt["candidate_answers"]:
        raise ValueError("Changed evidence inventory")
    return data


def independent(dev, held):
    dev_answers = {aid for row in dev["queries"] for aid in row["answers"]}
    dev_sources = {dev["documents"][str(aid)]["qid"] for aid in dev_answers}
    dev_titles = [set(tokens(row["title"])) for row in dev["queries"]]
    result = []
    dropped = Counter()
    for query in held["queries"]:
        answers = query["answers"]
        title = set(tokens(query["title"]))
        if dev_answers.intersection(answers) or any(held["documents"][str(aid)]["qid"] in dev_sources for aid in answers):
            dropped["shared_gold_source"] += 1
        elif any(title and previous and len(title & previous) / len(title | previous) >= .80
                 for previous in dev_titles):
            dropped["near_duplicate_title"] += 1
        else:
            result.append(query)
    if not result:
        raise ValueError("No independent queries")
    return dict(held, queries=result), dict(dropped)


def index(data, encoder):
    ids = sorted(map(int, data["documents"]))
    records = [data["documents"][str(i)] for i in ids]
    vector = encoder.encode([f"{r['question']}\n{r['question_text']}\n{r['answer']}" for r in records])
    owner = [r["owner"] for r in records]
    occurrences = Counter(term for r in records for term in set(tokens(r["answer"])))
    idf = defaultdict(float, {word: math.log((1 + len(records))/(1 + count))
                              for word, count in occurrences.items()})
    return vector, records, owner, dict(zip(ids, range(len(ids)), strict=True)), idf


def work(query, corpus, encoder):
    vector, records, owner, ids, _ = corpus
    text = f"{query['title']}\n{query['query']}"
    order, first, budget = shortlist(text, vector, owner, encoder)
    added = checks("same_owners", order, owner, first, [0] * len(first), budget)
    gold = {ids[aid] for aid in query["answers"]}
    return text, first, added, gold


def pair_windows(record, terms, idf):
    full = record["answer"]
    lead = full[:WINDOW_CHARS]
    target, _ = select_window(full, terms, idf)
    if len(lead) != len(target):
        raise ValueError("A routed view must keep the lead character budget")
    lead_terms, target_terms = set(tokens(lead)), set(tokens(target))
    gain = math.fsum(idf[t] for t in terms & target_terms) - math.fsum(idf[t] for t in terms & lead_terms)
    weight = math.fsum(idf.get(t, 0) for t in terms)
    title = record["question"]
    return f"{title}\n{lead}", f"{title}\n{target}", gain / weight if weight else 0.0


def score(reader, text, windows, split, instruction):
    scores, capped = [], 0
    for group in (windows[:split], windows[split:]):
        if group:
            values, clipped = reader.score_pairs([(text, doc) for doc in group], task=instruction)
            if not np.isfinite(values).all():
                raise ValueError("Invalid reader output")
            scores.extend(values)
            capped += clipped
    return scores, capped


def fit(data, corpus, encoder, reader):
    _, records, owner, _, idf = corpus
    rows = []
    for query in data["queries"]:
        text, first, added, gold = work(query, corpus, encoder)
        text = text[:REQUEST_CHARS]
        documents = first + added
        terms = set(tokens(text))
        lead, target, gains = zip(*(pair_windows(records[doc], terms, idf) for doc in documents), strict=True)
        left, lclip = score(reader, text, lead, len(first), TASK)
        right, rclip = score(reader, text, target, len(first), TASK)
        if lclip or rclip:
            raise ValueError("Development short windows unexpectedly reached the token cap")
        rows.append((documents, gold, left, right, gains))
    grid = sorted({0.0, *(x for *_, gains in rows for x in gains if x > 0)})
    best = None
    for threshold in grid:
        hits, switches = 0, 0
        for docs, gold, lead, target, gains in rows:
            keep = [gain > threshold for gain in gains]
            values = [y if use_target else x for x, y, use_target in zip(lead, target, keep, strict=True)]
            hits += evaluate(docs, values, owner, gold)["retained"]
            switches += sum(keep)
        current = (hits, -switches, threshold)
        if best is None or best < current:
            best = current
    reference = sum(evaluate(docs, lead, owner, gold)["retained"] for docs, gold, lead, _, _ in rows)
    return best[-1], dict(queries=len(rows), hits=best[0], lead_hits=reference, switched=-best[1],
                          candidate_cuts=len(grid))


def test(data, corpus, encoder, reader, threshold, policies=POLICIES):
    _, records, owner, _, idf = corpus
    outputs, times = defaultdict(list), defaultdict(list)
    queries = data["queries"]
    for i, query in enumerate([queries[0], *queries]):
        text, first, added, gold = work(query, corpus, encoder)
        documents = first + added
        offset = i % len(policies)
        for policy in policies[offset:] + policies[:offset]:
            clock = perf_counter()
            elongated = policy in ("more_request", "both_changes")
            explicit = policy in ("explicit_prompt", "both_changes")
            request = text[:EXTENDED_CHARS if elongated else REQUEST_CHARS]
            changed_window = 0
            if policy in ("targeted", "routed"):
                terms = set(tokens(request))
                windows = []
                for doc in documents:
                    lead, target, gain = pair_windows(records[doc], terms, idf)
                    chosen = target if policy == "targeted" or gain > threshold else lead
                    windows.append(chosen)
                    changed_window += chosen != lead
            else:
                windows = [f"{records[doc]['question']}\n{records[doc]['answer'][:WINDOW_CHARS]}" for doc in documents]
            margins, capped = score(reader, request, windows, len(first), EXPLICIT if explicit else TASK)
            outcome = evaluate(documents, margins, owner, gold)
            if i:
                outputs[policy].append(dict(**outcome, changed_window=changed_window, capped=capped,
                                            extended_request=int(elongated and len(text) > REQUEST_CHARS)))
                times[policy].append((perf_counter() - clock) * 1000)
    groups = connected_groups(queries)
    baseline = outputs["lead"]
    result = {}
    for name, rows in outputs.items():
        totals = {key: sum(row[key] for row in rows) for key in rows[0]}
        pairs = list(zip(rows, baseline, strict=True))
        uncapped_extension = [a["retained"] - b["retained"] for a, b in pairs
                              if a["extended_request"] and not (a["capped"] or b["capped"])]
        result[name] = dict(
            uncapped_extensions=len(uncapped_extension), uncapped_extension_net_hits=sum(uncapped_extension),
            totals=totals, p95_ms=float(np.quantile(times[name], .95)),
            recovered=sum(a["retained"] > b["retained"] for a,b in pairs),
            lost=sum(a["retained"] < b["retained"] for a,b in pairs),
            hit_delta=clustered_paired_interval(
                [a["retained"] - b["retained"] for a,b in pairs], groups,
            ),
        )
    return result


def run(args):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise ValueError("Require Slurm and a fresh output")
    prior = json.loads(args.anchor.read_bytes())
    dev = load(args.dev, args.dev_manifest)
    candidate = load(args.held, args.held_manifest)
    holdout, excluded = independent(dev, candidate)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    models = dict(embedding=verify_checkpoint(args.embedding)["revision"], reranker=reader.provenance["revision"])
    if any(prior["revisions"][name] != revision for name, revision in models.items()):
        raise ValueError("Changed checkpoint")
    development_index = index(dev, encoder)
    threshold, fit_summary = fit(dev, development_index, encoder, reader)
    if fit_summary["lead_hits"] != prior["arms"]["0.6b_lead"]["totals"]["retained"]:
        raise ValueError("Development lead baseline did not reproduce")
    live = test(dev, development_index, encoder, reader, threshold, policies=("lead", "routed"))
    fit_summary["live_hits"] = live["routed"]["totals"]["retained"]
    report = test(holdout, index(holdout, encoder), encoder, reader, threshold)
    write(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], anchor_sha256=digest(args.anchor),
        code_sha256=digest(Path(__file__)), held_manifest=digest(args.held_manifest),
        holdout_queries=len(holdout["queries"]), excluded=excluded, threshold=threshold,
        development=fit_summary, reports=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "dev_manifest", "held", "held_manifest", "anchor", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
