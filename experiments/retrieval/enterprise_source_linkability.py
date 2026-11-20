"""Audit exact locator-shaped fields and named role cues in enterprise search."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from experiments.retrieval import enterprise_declaration_reach as declaration
from experiments.retrieval import enterprise_project_role_rescue as project
from experiments.retrieval.enterprise_routed_review import digest, rows, write


PATENT = re.compile(
    r"(?<![A-Za-z0-9])(?:CN|US|EP|WO|ZL)[ \-]*\d{7,13}(?:[.．]\d|[A-Z]\d?)?(?![A-Za-z0-9])"
    r"|(?:\u7533\u8bf7\u53f7|\u516c\u5f00\u53f7|\u6388\u6743\u53f7|\u4e13\u5229\u53f7)"
    r"\s*[:\uff1a]?\s*\d{7,13}(?:[.．][\dX])?(?![A-Za-z0-9])",
    re.I,
)
PROJECT = re.compile(
    r"(?:\u9879\u76ee\u7f16\u53f7|\u8bfe\u9898\u7f16\u53f7|\u7acb\u9879\u7f16\u53f7|"
    r"\u8d44\u52a9\u7f16\u53f7|\u57fa\u91d1\u7f16\u53f7|grant (?:number|no\.?))"
    r"\s*[:\uff1a]?\s*([A-Za-z0-9][A-Za-z0-9._/-]{3,29})(?![A-Za-z0-9])",
    re.I,
)
NAMED = frozenset(("named_inventor", "role_and_exact_name"))


def locators(field, text):
    pattern = PATENT if field == "patent" else PROJECT if field == "project" else None
    if pattern is None:
        return []
    return [
        (match.start(), match.end()) for match in pattern.finditer(text)
        if sum(c.isdigit() for c in match.group()) >= 3
    ]


def screen(field, original, role_cues):
    ID_spans = locators(field, original)
    named = [(left, right) for left, right, label in role_cues if label in NAMED]
    same_clause = [(left, right) for left, right in named
                   if any(left <= a < b <= right for a, b in ID_spans)]
    return dict(field=field, locators=ID_spans, named=named, same_clause=same_clause)


def source_counts(records):
    counts = defaultdict(Counter)
    for item in records.values():
        counts[item["field"]].update(records=1, with_locator=bool(item["locators"]),
                                     with_named_role_cue=bool(item["named"]),
                                     with_both_in_field=bool(item["locators"] and item["named"]),
                                     with_both_in_clause=bool(item["same_clause"]))
    return counts


def prepare(dataset, corpus, output):
    manifest_path = dataset / "manifest.json"
    metadata = json.loads(manifest_path.read_bytes())
    owners = project.profile_subset(dataset, "experts", metadata)
    raw = project.profile_subset(dataset, "corpus", metadata)
    prepared = {}
    for row in rows(corpus):
        source = raw[row["id"]]
        if (source["expert_id"], source["text"]) != (row["expert_id"], row["text"]):
            raise ValueError("Profile or ownership changed")
        field = (declaration.FIELD_KIND.get(source.get("field"), "other_profile")
                 if source["kind"] == "profile" else "paper")
        name = project.name_key(owners[row["expert_id"]]["name"] or "")
        spans = declaration.role_spans(source, name)
        prepared[row["id"]] = screen(field, source["text"], spans)
    if prepared.keys() != raw.keys():
        raise ValueError("Incomplete enterprise corpus")
    write(output, dict(
        source_sha256=digest(corpus), manifest_sha256=digest(manifest_path),
        flags=prepared, counts=source_counts(prepared),
    ))


class Linkability:
    def __init__(self, projection, corpus_path, sources):
        report = json.loads(projection.read_bytes())
        self.flags = report["flags"]
        if digest(corpus_path) != report["source_sha256"] or self.flags.keys() != sources.keys():
            raise ValueError("Changed localization input")
        for source, item in self.flags.items():
            length = len(sources[source]["text"])
            for kind in ("locators", "named", "same_clause"):
                if any(not 0 <= left < right <= length for left, right in item[kind]):
                    raise ValueError("Invalid original text offsets")
        self.by_owner = defaultdict(set)
        for doc, item in self.flags.items():
            if item["same_clause"]:
                self.by_owner[sources[doc]["expert_id"]].add(doc)
        self.corpus_counts = report["counts"]
        self.panels = defaultdict(Counter)

    def audit(self, arm, people, excerpts, scores, issued):
        counts = self.panels[arm]
        counts.update(requests=1, candidates=len(people))
        reachable = False
        catalog_reachable = False
        for owner in people:
            has_binding = False
            topic = False
            for doc, left, right in excerpts[owner]:
                item = self.flags[doc]
                local = {kind: any(left <= a < b <= right for a, b in item[kind])
                         for kind in ("locators", "named", "same_clause")}
                counts.update(excerpts=1, locator_in_source=bool(item["locators"]),
                              locator_in_excerpt=local["locators"], name_in_excerpt=local["named"],
                              coanchor_in_excerpt=local["same_clause"])
                has_binding |= local["same_clause"]
                topic |= scores[owner, (doc, left, right), 0] > 0
            has_catalog_binding = bool(self.by_owner.get(owner))
            counts.update(candidates_with_coanchor=has_binding,
                          candidates_with_topic_and_coanchor=has_binding and topic,
                          candidates_with_coanchor_anywhere=has_catalog_binding,
                          candidates_with_topic_and_coanchor_anywhere=has_catalog_binding and topic)
            catalog_reachable |= has_catalog_binding
            reachable |= has_binding
        counts["requests_with_coanchor"] += reachable
        counts["requests_with_coanchor_anywhere"] += catalog_reachable
        for record in issued:
            counts["provisionals"] += 1
            if any(any(left <= a < b <= right for a, b in self.flags[doc]["same_clause"])
                   for doc, left, right in record["citations"]):
                counts["provisionals_with_coanchor_citation"] += 1

    def aggregate(self):
        return dict(inventory=self.corpus_counts, search=self.panels, code_sha256=digest(Path(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "corpus", "output"):
        parser.add_argument(name, type=Path)
    prepare(**vars(parser.parse_args()))
