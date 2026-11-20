"""Audit which publisher sources are reachable after one additional declaration read."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from expert_search.statistics import clustered_paired_interval
from experiments.credit import cross_source_search as search
from experiments.credit.source_confirmation import accepts, calibrate, measure, notice_kind, ordered


STYLES = ("title_sketch", "gapped_title_sketch", "edited_abstract", "abridged_abstract")
ARMS = ("next_bm25", "next_ordered", "owner_first", "owner_ordered")
POOL = 8


def source_clue(document, style):
    if style == "gapped_title_sketch":
        terms = search.normalized(document[0]).split()
        if len(terms) >= 7:
            start = (len(terms) - 7) // 2
            return " ".join(terms[start:start + 7:2])
    return search.lead(document, "title_sketch" if style == "gapped_title_sketch" else style)


class Catalogue:
    def __init__(self, documents, ownership):
        self.documents = documents
        self.ownership = ownership
        self.ids = sorted(documents)
        self.segments = {
            doi: [search.normalized(text).split() for text in (title, *search.sentences(abstract))]
            for doi, (title, abstract) in documents.items()
        }
        self.kinds = {doi: notice_kind(title) for doi, (title, _) in documents.items()}
        self.byline = {
            doi: {author["person"] for author in info["authors"] if author["person"]}
            for doi, info in ownership.items()
        }
        with TemporaryDirectory() as scratch:
            texts = ["\n".join(documents[doi]) for doi in self.ids]
            if len(set(texts)) != len(texts):
                raise ValueError("Duplicate searchable documents")
            self.lexical = search.build_lexical({text: i for i, text in enumerate(texts)}, Path(scratch))

    def rank(self, clue, *, excluded=()):
        docs = self.lexical.document_scores(clue)
        kind = notice_kind(clue)
        ids = [self.ids[int(i)] for i in sorted(docs, key=lambda i: (-docs[i], self.ids[int(i)]))
               if self.ids[int(i)] not in excluded and self.kinds[self.ids[int(i)]] == kind]
        return ids[:POOL]

    def literal(self, clue, excluded):
        terms = search.normalized(clue)
        if not terms:
            return None
        snippet = f" {terms} "
        candidates = [
            doi for doi, pair in self.documents.items()
            if doi not in excluded and self.kinds[doi] == notice_kind(clue)
            and any(snippet in f" {search.normalized(text)} " for text in pair)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def confidence(self, clue, ranking):
        terms = search.normalized(clue).split()
        return measure(terms, ranking, self.segments, "ordered")

    def proposals(self, clue, ranking, people):
        candidates = ranking[1:]
        if not candidates:
            return dict.fromkeys(ARMS)
        terms = search.normalized(clue).split()
        by_order = sorted(candidates, key=lambda doi: -max(ordered(terms, text) for text in self.segments[doi]))
        owned = [doi for doi in candidates if self.byline[doi] & people]
        owned_order = [doi for doi in by_order if self.byline[doi] & people]
        return dict(next_bm25=candidates[0], next_ordered=by_order[0],
                    owner_first=(owned or candidates)[0], owner_ordered=(owned_order or by_order)[0])


def load_fixture(fixture, folders):
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        if search.digest(fixture / name) != expected:
            raise ValueError(f"Changed fixture: {name}")
    queries = json.loads((fixture / "queries.json").read_bytes())
    documents = search.searchable_text(folders)
    ownership = search.load(folders)
    expected = {source["doi"]: source["source_sha256"] for source in manifest["input_sources"]}
    if {doi: s["source_sha256"] for doi, s in ownership.items()} != expected or ownership.keys() != documents.keys():
        raise ValueError("Changed XML inputs")
    return Catalogue(documents, ownership), queries


def fit(catalogue, queries):
    targets = {row[side] for row in queries for side in ("left", "right")}
    policies = {}
    for style in STYLES:
        examples = []
        for doi in sorted(catalogue.documents.keys() - targets):
            clue = source_clue(catalogue.documents[doi], style)
            for hidden in (False, True):
                ranking = catalogue.rank(clue, excluded=(doi,) if hidden else ())
                examples.append((catalogue.confidence(clue, ranking), None if hidden else doi))
        policies[style] = calibrate(examples)
    return policies, len(catalogue.documents.keys() - targets)


def try_case(catalogue, case, reverse, style, policy, missing):
    first, wanted = (case["right"], case["left"]) if reverse else (case["left"], case["right"])
    role_a, role_b = (case["right_role"], case["left_role"]) if reverse else (case["left_role"], case["right_role"])
    first_role = search.eligible(catalogue.ownership[first], role_a)
    people = {row["person"] for row in first_role if row["person"]}
    clue = source_clue(catalogue.documents[wanted], style)
    excluded = (first, wanted) if missing else (first,)
    ranking = catalogue.rank(clue, excluded=excluded)
    initial = ranking[0] if ranking else None
    confirmation = catalogue.confidence(clue, ranking)
    confirmed = confirmation[0] if accepts(confirmation, policy) else None
    proposals = catalogue.proposals(clue, ranking, people)
    truth = not missing and bool(case["gold"])

    def supports(doi):
        return bool(doi and any(row["person"] in people
                                for row in search.eligible(catalogue.ownership[doi], role_b)))

    assert supports(wanted) == bool(case["gold"])
    initial_true = truth and initial == wanted and supports(initial)
    literal = catalogue.literal(clue, excluded)
    source = literal or confirmed
    source_first = dict(reads=int(source is not None), literal_confirmed=int(literal is not None),
                        correct_links=int(truth and source == wanted and supports(source)),
                        wrong_links=int(source != wanted and supports(source)))
    # Evaluation reveals declarations only for the inspected sources.
    observed_links = {doi: supports(doi) for doi in {initial, *proposals.values()}}
    outcomes = {}
    for arm, second in proposals.items():
        inspected = {doi for doi in (initial, second) if doi is not None}
        links = {doi for doi in inspected if observed_links[doi]}
        allowed = links & {confirmed}
        raw_correct = truth and wanted in links
        outcomes[arm] = dict(
            reads=len(inspected), switched=int(second != proposals["next_bm25"]),
            raw_correct=int(raw_correct), extra_correct=int(raw_correct and not initial_true),
            raw_wrong=int(any(doi != wanted or not truth for doi in links)),
            gated_correct=int(truth and wanted in allowed),
            gated_wrong=int(any(doi != wanted or not truth for doi in allowed)),
        )
    opportunity = dict(
        first_valid=int(initial_true), positive_miss=int(truth and not initial_true),
        reachable=int(truth and not initial_true and wanted in ranking[1:]),
        reachable_and_confirmed=int(truth and not initial_true and wanted in ranking[1:] and confirmed == wanted),
    )
    pair = "|".join(sorted((first, wanted)))
    return pair, opportunity, outcomes, source_first


def summarize(records, intervals=True):
    totals = {arm: Counter() for arm in ARMS}
    opportunities = Counter()
    source_first = Counter()
    for _, operation, outcomes, one_read in records:
        source_first.update(one_read)
        opportunities.update(operation)
        for arm, values in outcomes.items():
            totals[arm].update(values)
    result = dict(requests=len(records), publication_pairs=len({row[0] for row in records}),
                  opportunity=dict(opportunities), source_first=dict(source_first),
                  arms={arm: dict(values) for arm, values in totals.items()})
    if intervals:
        for arm in ARMS[1:]:
            result["arms"][arm]["paired_extra_delta_vs_next_bm25"] = clustered_paired_interval(
                [r[2][arm]["extra_correct"] - r[2]["next_bm25"]["extra_correct"] for r in records],
                [r[0] for r in records],
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    catalogue, queries = load_fixture(args.fixture, args.sources)
    policies, calibration_titles = fit(catalogue, queries)
    groups = {}
    for style in STYLES:
        groups[style] = {
            label: summarize([
                try_case(catalogue, case, reverse, style, policies[style], missing)
                for case in queries if case["kind"] in kinds
                for reverse in (False, True)
            ], intervals=not missing and kinds == ("positive",))
            for label, missing, kinds in (
                ("both_visible_positive", False, ("positive",)),
                ("both_visible_no_link", False, ("no_link",)),
                ("target_withheld", True, ("positive",)),
            )
        }
    with args.output.open("x") as handle:
        json.dump(dict(
            fixture_sha256=search.digest(args.fixture / "manifest.json"), sources=len(catalogue.ids),
            script_sha256=search.digest(Path(__file__)), policies=policies, calibration_titles=calibration_titles,
            pool=POOL, groups=groups,
        ), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
