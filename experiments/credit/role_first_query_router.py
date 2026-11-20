"""Learn a small pre-verification nomination gate and test it across catalogs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.credit import role_first_query_equal_reads as live
from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("topic_fixed", "person_fixed")
FEATURES = ("none", "word_overlap", "all_words")


def features(catalog, topic, pair, ranked, visible):
    prefix = set(ranked[:live.FIRST])
    role_posting = catalog.person[pair]
    selected = [i for i in ranked if i not in visible or i in role_posting][:live.FIRST]
    matching_words = set(pilot.tokens(topic))
    best = dict.fromkeys(FEATURES, len(ranked) + 1)
    introduced = set(selected) - prefix
    for rank, index in enumerate(ranked, 1):
        if index not in introduced or index not in visible or index not in role_posting:
            continue
        title_words = set(pilot.tokens(catalog.texts[index].split("\n", 1)[0]))
        best["none"] = min(best["none"], rank)
        if matching_words & title_words:
            best["word_overlap"] = min(best["word_overlap"], rank)
        if matching_words and matching_words <= title_words:
            best["all_words"] = min(best["all_words"], rank)
    return dict(exclusions=len((prefix & visible) - role_posting), **best)


def choose(rule, row):
    if rule["feature"] == "never":
        return "topic_fixed"
    if row["exclusions"] >= rule["exclusions"] and row[rule["feature"]] <= rule["max_rank"]:
        return "person_fixed"
    return "topic_fixed"


def collect(catalog, queries, visible, cutoff, rankings, score_cache, *, arms=ARMS):
    rows = []
    for topic, _, pair in queries:
        ranked = rankings[topic]

        def score(query, sources):
            key = (query, tuple(sources))
            if key not in score_cache:
                score_cache[key] = catalog.verify(query, sources)
            return score_cache[key]

        proxy = SimpleNamespace(rank=lambda query: ranked, verify=score, person=catalog.person,
                                bindings=catalog.bindings, topics=catalog.topics)
        row = features(catalog, topic, pair, ranked, visible)
        row["topic"] = topic
        for arm in arms:
            result = live.replay(proxy, topic, pair, visible, cutoff, arm)
            row[arm] = {key: result[key] for key in ("grounded", "off_label", "issued", "reads")}
        rows.append(row)
    return rows


def enumerate_rules():
    yield dict(feature="never", exclusions=0, max_rank=0)
    for feature in FEATURES:
        for excluded in (1, 2, 3):
            for rank in (6, 8, 12, 20):
                yield dict(feature=feature, exclusions=excluded, max_rank=rank)


def summarize(rows, selector):
    totals = dict.fromkeys(("grounded", "off_label", "issued", "reads", "person_routes"), 0)
    for row in rows:
        action = selector(row)
        for key, value in row[action].items():
            totals[key] += value
        totals["person_routes"] += action == "person_fixed"
    return totals


def fit(rows):
    off_limit = sum(row["topic_fixed"]["off_label"] for row in rows)
    choices = []
    for rule in enumerate_rules():
        result = summarize(rows, lambda row: choose(rule, row))
        if result["off_label"] <= off_limit:
            choices.append((rule, result))
    best, metrics = min(choices, key=lambda item: (-item[1]["grounded"], item[1]["off_label"],
                                                    item[1]["person_routes"]))
    return best, metrics, len(choices)


def panels(catalog, publisher):
    if publisher == "elife":
        return factorial.panel_queries(catalog)
    selected = sorted(topic for topic, articles in catalog.topics.items() if len(articles) >= 4)
    return {"leaf_topics": (catalog.topics, list(pilot.query_panel(catalog, selected)))}


def run_panels(catalog, publisher, cutoff, *, arms=ARMS):
    selected = panels(catalog, publisher)
    topic_names = {topic for _, queries in selected.values() for topic, _, _ in queries}
    rankings = {topic: catalog.rank(topic) for topic in sorted(topic_names)}
    complete = {i for i, row in enumerate(catalog.articles)
                if row.get("declaration_complete", True)}
    cache, dev, test = {}, [], {}
    for field, (lookup, queries) in selected.items():
        catalog.topics = lookup
        for mask, visible in catalog.visible.items():
            rows = collect(catalog, queries, visible & complete, cutoff, rankings, cache, arms=arms)
            valid = []
            for row in rows:
                if publisher == "elife" and int.from_bytes(
                    pilot.stable("factorial-calibration-v1", row["topic"])[:8], "big",
                ) % 3 == 0:
                    dev.append(row)
                else:
                    valid.append(row)
            test[f"{field}:{mask}"] = valid
    return dev, test


def compare(test, rule):
    output = {}
    for cell, rows in test.items():
        metrics = {arm: summarize(rows, lambda row, action=arm: action) for arm in ARMS}
        def select(row):
            return choose(rule, row)

        metrics["router"] = summarize(rows, select)
        paired = {}
        for baseline in ARMS:
            paired[baseline] = {}
            for field in ("grounded", "off_label"):
                values = [(row[select(row)][field] - row[baseline][field]) / pilot.TOP for row in rows]
                ci = pilot.clustered_paired_interval(values, [row["topic"] for row in rows])
                ci["sampling_unit"] = "whole topic; query-weighted mean"
                paired[baseline][field] = ci
        output[cell] = dict(queries=len(rows), arms=metrics, router_vs=paired)
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("elife", "plos", "plos_manifest", "reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    prior = json.loads(args.reference.read_bytes())
    plos_pin = json.loads(args.plos_manifest.read_bytes())
    if digest(args.elife) != prior["projection_sha256"] or digest(args.plos) != plos_pin["projection_sha256"]:
        raise ValueError("Unexpected publisher projection")
    for directory, key in ((args.embedding, "embedding_sha256"),
                           (args.reranker / factorial.Verifier.name, "reranker_sha256")):
        if {name: digest(directory / name) for name in prior[key]} != prior[key]:
            raise ValueError("Unexpected model checkpoint")

    encoder, verifier = factorial.Encoder(args.embedding), factorial.Verifier(args.reranker)
    elife = pilot.Catalog(json.loads(args.elife.read_bytes())["sources"], encoder, verifier)
    dev, elife_test = run_panels(elife, "elife", prior["cutoff"])
    rule, development, feasible = fit(dev)
    plos = pilot.Catalog(json.loads(args.plos.read_bytes())["sources"], encoder, verifier)
    _, plos_test = run_panels(plos, "plos", prior["cutoff"])
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.reference),
        elife_sha256=digest(args.elife), plos_sha256=digest(args.plos),
        chosen_rule=rule, development=dict(rows=len(dev), feasible_rules=feasible, chosen=development,
                                          topic_fixed=summarize(dev, lambda row: "topic_fixed")),
        elife=compare(elife_test, rule), plos=compare(plos_test, rule),
    ))


if __name__ == "__main__":
    main()
