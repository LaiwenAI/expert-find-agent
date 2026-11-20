"""Allocate a fixed number of costly checks across ready topics."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.followir.evaluate_same_passage_routing import cutoff, decisions, summarize
from expert_search.statistics import paired_interval


BLOCK_SIZE = 5
DIVISOR = 4


def pick(groups, best, *, surplus):
    """Keep one review per eligible topic, then merge upper-bounded priorities."""
    local = [len(group) // DIVISOR for group in groups]
    total = sum(local)
    ranked = [sorted(group, key=best) for group in groups]
    if not surplus:
        return [i for group, quota in zip(ranked, local, strict=True) for i in group[:quota]]
    reserved = [group[0] for group, quota in zip(ranked, local, strict=True) if quota]
    eligible = [i for group, quota in zip(ranked, local, strict=True) if quota
                for i in group[1:quota + surplus]]
    return reserved + sorted(eligible, key=best)[:total - len(reserved)]


def key(row):
    return sha256(f"same-passage-v1:{row['topic']}:{row['key']}".encode()).digest()


def priorities(rows, cheap_threshold, kind):
    distance = [abs(row["cheap"] - cheap_threshold) for row in rows]
    ranks = [key(row) for row in rows]
    if kind == "hash":
        return lambda i: (ranks[i],)
    if kind == "boundary":
        return lambda i: (distance[i], ranks[i])
    raise ValueError(f"Unknown routing priority: {kind}")


def blocks(topics, offset):
    sequence = list(topics.values())
    sequence = sequence[offset:] + sequence[:offset]
    return [sequence[start:start + BLOCK_SIZE] for start in range(0, len(sequence), BLOCK_SIZE)]


def choose(rows, topics, cheap_threshold, offset, method, surplus=1):
    if method == "local":
        return pick(list(topics.values()), priorities(rows, cheap_threshold, "boundary"), surplus=0)
    priority = priorities(rows, cheap_threshold, "hash" if method == "pool_hash" else "boundary")
    return [i for block in blocks(topics, offset) for i in pick(block, priority, surplus=surplus)]


def comparison(rows, accepted, always_large, topics):
    positive = np.asarray([row["label"] == "stable_positive" for row in rows])
    invalid = np.asarray([row["label"] == "invalidated" for row in rows])
    return {
        "stable_positive_shared_with_4b": int((accepted & always_large & positive).sum()),
        "stable_positive_only_route": int((accepted & ~always_large & positive).sum()),
        "stable_positive_only_4b": int((always_large & ~accepted & positive).sum()),
        "positive_document_delta_per_topic": paired_interval([
            int((accepted[group] & positive[group]).sum()) - int((always_large[group] & positive[group]).sum())
            for group in topics.values()
        ]),
        "invalidated_document_delta_per_topic": paired_interval([
            int((accepted[group] & invalid[group]).sum()) - int((always_large[group] & invalid[group]).sum())
            for group in topics.values()
        ]),
    }


def evaluate(rows, thresholds):
    small, large, gain = decisions(rows, thresholds)
    topics = defaultdict(list)
    for i, row in enumerate(rows):
        topics[row["topic"]].append(i)

    def report(selected):
        accepted = small.copy()
        accepted[selected] = large[selected]
        counts = Counter(rows[i]["topic"] for i in selected)
        return dict(summarize(rows, accepted, small, selected, topics),
                    vs_always_4b=comparison(rows, accepted, large, topics),
                    calls_per_topic=Counter(counts[topic] for topic in topics),
                    topics_with_positive=sum(any(accepted[i] and rows[i]["label"] == "stable_positive"
                                                 for i in group) for group in topics.values()))

    result = {}
    local = choose(rows, topics, thresholds[0], 0, "local")
    result["local"] = report(local)
    result["pooled"] = []
    for offset in range(BLOCK_SIZE):
        row = {}
        for surplus in (1, 2):
            for name in ("pool_boundary", "pool_hash"):
                selected = choose(rows, topics, thresholds[0], offset, name, surplus)
                if len(selected) != len(local) or len(set(selected)) != len(selected):
                    raise ValueError("Pooled allocation failed to keep the total call count")
                row[f"{name}_plus_{surplus}"] = report(selected)
        result["pooled"].append(row)

    selected = set(local)
    missed = [i for i, row in enumerate(rows) if row["label"] == "stable_positive"
              and large[i] and not small[i] and i not in selected]
    rank = {}
    best = priorities(rows, thresholds[0], "boundary")
    for topic in topics.values():
        rank.update({i: r for r, i in enumerate(sorted(topic, key=best), start=1)})
    caps = {i: len(group) // DIVISOR for group in topics.values() for i in group}
    result["misses_after_local"] = {
        "unreviewed_4b_only_positives": len(missed),
        "unreviewed_ranks": sorted(rank[i] for i in missed),
        "within_relaxed_rank_cap": {
            str(extra): sum(caps[i] > 0 and rank[i] <= caps[i] + extra for i in missed)
            for extra in (1, 2)
        },
        "local_replacements_removed_small_only_positives": sum(
            int(rows[i]["label"] == "stable_positive" and gain[i] < 0) for i in local
        ),
    }
    return result


def run(scores, reference, output):
    source = json.loads(scores.read_bytes())
    prior = json.loads(reference.read_bytes())
    if source["source_sha256"] != prior["source_sha256"] or source["code_sha256"] != prior["score_script_sha256"]:
        raise ValueError("Different baseline projection")
    panels = {name: values["rows"] for name, values in source["panels"].items()}
    development = {k: panels[k] for k in ("core", "news")}
    thresholds = tuple(cutoff(development, name) for name in ("cheap", "costly"))
    verified = prior["result"]["transferred_risk_cutoffs"]
    if list(thresholds) != verified["cutoffs"]:
        raise ValueError("Different cutoff")
    report = evaluate(panels["robust"], thresholds)
    local = report["local"]
    baseline = verified["budgets"]["1/4"]["boundary"]
    if any(local[k] != baseline[k] for k in baseline):
        raise ValueError("Local route failed to reproduce its published trace")
    with output.open("x") as stream:
        json.dump(dict(report=report, cutoffs=thresholds,
                       score_sha256=sha256(scores.read_bytes()).hexdigest(),
                       reference_sha256=sha256(reference.read_bytes()).hexdigest(),
                       source_sha256=source["source_sha256"],
                       script_sha256=sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scores", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
