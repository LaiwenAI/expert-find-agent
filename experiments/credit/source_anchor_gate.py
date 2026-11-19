"""Test independent lexical source checks with and without author-linked search."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log
from pathlib import Path
from time import perf_counter_ns

from expert_search.retrieval import tokens
from experiments.credit.scoped_credit_search import STOPS
from experiments.credit.scoped_source_confirmation import MISSING, VIEWS, graph


def content_terms(text):
    return {word for word in tokens(text) if len(word) > 3 and word not in STOPS}


class AnchorIndex:
    def __init__(self, documents):
        self.postings = defaultdict(set)
        self.owners = {doi: set(row["identities"]) for doi, row in documents.items()}
        for doi, row in documents.items():
            for word in content_terms(row["index"]):
                self.postings[word].add(doi)
        self.weights = {word: log((1 + len(documents)) / len(ids))
                        for word, ids in self.postings.items()}

    def match(self, terms, absent):
        scores = Counter()
        unique = Counter()
        for word in sorted(terms):
            candidates = self.postings.get(word, ())
            for doi in candidates:
                if doi != absent:
                    scores[doi] += self.weights[word]
                    unique[doi] += len(candidates) == 1
        ranked = sorted(scores, key=lambda doi: (-scores[doi], doi))
        return ranked, unique

    def linked(self, left, right):
        candidates = [
            (i + j, max(i, j), a, b)
            for i, a in enumerate(left[:12], 1)
            for j, b in enumerate(right[:12], 1)
            if a != b and min(i, j) <= 4 and self.owners[a] & self.owners[b]
        ]
        return min(candidates)[2:] if candidates else None

    def query(self, left, right, absent):
        ranked_a, unique_a = self.match(left, absent)
        ranked_b, unique_b = self.match(right, absent)
        independent = (ranked_a[0], ranked_b[0]) if ranked_a and ranked_b else None
        if independent and independent[0] == independent[1]:
            independent = None
        linked = self.linked(ranked_a, ranked_b)
        confirmed = linked if linked and all(
            votes[doi] >= 2 and sum(votes.values()) == votes[doi]
            for doi, votes in zip(linked, (unique_a, unique_b), strict=True)
        ) else None
        return {"independent": independent, "linked": linked, "linked_confirmed": confirmed}


def evaluate(documents, panel):
    index = AnchorIndex(documents)
    counts = defaultdict(Counter)
    durations = defaultdict(list)
    for left, right, _ in panel["pairs"]:
        pair = left, right
        for view in VIEWS:
            raw = tuple(content_terms(panel["clues"][doi][view]) for doi in pair)
            without_anchors = tuple({term for term in terms if len(index.postings.get(term, ())) != 1}
                                    for terms in raw)
            for variant, terms in (("original", raw), ("without_unique_terms", without_anchors)):
                for case in MISSING:
                    absent = {"present": None, "missing_left": left, "missing_right": right}[case]
                    start = perf_counter_ns()
                    answers = index.query(*terms, absent)
                    durations[view, variant].append((perf_counter_ns() - start) / 1e6)
                    for policy, answer in answers.items():
                        tally = counts[view, variant, policy]
                        tally["present" if absent is None else "missing"] += 1
                        tally["exact"] += absent is None and answer == pair
                        tally["wrong_present"] += absent is None and answer is not None and answer != pair
                        tally["wrong_missing"] += absent is not None and answer is not None
    return {
        "counts": {"/".join(key): tally for key, tally in counts.items()},
        "all_policies_p95_ms": {
            "/".join(key): sorted(values)[(95 * len(values) + 99) // 100 - 1]
            for key, values in durations.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.projection.read_bytes()
    panel = json.loads(args.panel.read_text())
    documents = json.loads(data)["documents"]
    if (panel["frozen_sha256"] != sha256(data).hexdigest()
            or panel["pairs"] != [list(pair) for pair in graph(documents)]):
        raise ValueError("Unexpected source inventory")
    with args.output.open("x") as output:
        json.dump(evaluate(documents, panel), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
