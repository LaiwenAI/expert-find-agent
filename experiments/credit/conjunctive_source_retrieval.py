"""Evaluate joint source recovery and same-author role evidence on PLOS records."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from experiments.credit.plos_role_inventory import declared_roles
from experiments.qasper.adaptive_evidence import digest
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


COLLECTION_SHA256 = "40f802d95aec45535c6b0427a87cb212ce7edbd355ab6030495358dd2be6acd7"
METHODS = ("bm25", "bm25_full", "dense", "dense_full",
           "exact_title_then_dense", "exact_title_or_agreement", "literal_or_agreement", "literal_then_bm25_full")
TASK = "Find the research paper described in this request."
REPEATS = 7
VIEWS = ("title", "abstract_body", "abstract_conclusion", "edited_conclusion")
ATTRIBUTION = ("same_author", "any_role", "paper_union")


def whitespace(text):
    return " ".join(text.split())


def sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", whitespace(text)) if part]


def edit_conclusion(text):
    substitutions = (
        (r"\bthis study\b", "the reported study"),
        (r"\bproposed method\b", "method under study"),
        (r"\bthe results\b", "these findings"),
        (r"\bdemonstrates\b", "shows"),
        (r"\bdemonstrate\b", "show"),
    )
    for pattern, replacement in substitutions:
        revised, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count:
            return revised
    return "The study concludes as follows: " + text


class ExcerptIndex:
    """Find a unique contiguous normalized quote through rare-token postings."""

    def __init__(self, abstracts):
        self.text = {}
        self.postings = defaultdict(list)
        for doi, abstract in abstracts.items():
            terms = tokens(abstract)
            self.text[doi] = " " + " ".join(terms) + " "
            for term in set(terms):
                self.postings[term].append(doi)

    def unique_source(self, lead):
        terms = tokens(lead)
        if not terms:
            return None
        candidates = min((self.postings.get(term, ()) for term in set(terms)), key=len)
        snippet = " " + " ".join(terms) + " "
        matches = [doi for doi in candidates if snippet in self.text[doi]]
        return matches[0] if len(matches) == 1 else None


def controls(fixture, source_dir):
    meta = json.loads((fixture / "manifest.json").read_bytes())
    query_path = fixture / "queries.json"
    if digest(query_path) != meta["rows_sha256"]:
        raise ValueError("Changed role inventory")
    source_map = {row["doi"]: row for row in meta["sources"]}
    case_groups = defaultdict(list)
    for row in json.loads(query_path.read_bytes()):
        case_groups[row["doi"]].append(row)
    queries, declarations = {}, {}
    for doi, role_rows in sorted(case_groups.items()):
        path = source_dir / f"pone.{doi.split('.')[-1]}.xml"
        if digest(path) != source_map[doi]["source_sha256"]:
            raise ValueError(f"Changed source {doi}")
        raw = path.read_bytes()
        memberships = declared_roles(raw, doi)
        if not memberships or any(not roles for roles in memberships):
            raise ValueError("Conjunctive cases require complete declarations")
        pos = {r["requested_role"]: set(r["matching_author_ordinals"]) for r in role_rows}
        for role, expected in pos.items():
            if expected != {i for i, roles in enumerate(memberships, 1) if role in roles}:
                raise ValueError("Publisher role labels disagree")
        declarations[doi] = tuple(frozenset(roles) for roles in memberships)
        doc = ET.fromstring(raw)
        bits = doc.findall("./front/article-meta/abstract//p")
        abstract = whitespace(" ".join("".join(node.itertext()) for node in bits))
        parts = sentences(abstract)
        if len(parts) < 3:
            raise ValueError(f"Abstract is too short for non-overlapping query views: {doi}")
        queries[doi] = {
            "title": role_rows[0]["publication_title"],
            "abstract_body": " ".join(" ".join(parts[1:-1]).split()[:70]),
            "abstract_conclusion": parts[-1],
            "edited_conclusion": edit_conclusion(parts[-1]),
            "pairs": [(first, second, sorted(pos[first] & pos[second]))
                      for first, second in combinations(sorted(pos), 2)],
        }
    return queries, declarations, meta["rows_sha256"]


def prepare_collection(path, cases):
    if digest(path) != COLLECTION_SHA256:
        raise ValueError("Changed PLOS Search snapshot")
    source = json.loads(path.read_bytes())["response"]
    indexed, abstracts = {}, {}
    for row in source["docs"]:
        title = whitespace(row["title"])
        abstract = " ".join(row["abstract"])
        abstracts[row["id"]] = abstract
        intro = next(iter(sentences(abstract)), "")
        indexed[row["id"]] = (title, title + "\n" + intro, title + "\n" + whitespace(abstract))
    if len(indexed) != source["numFound"] or not set(cases) <= indexed.keys():
        raise ValueError("Missing or duplicate papers in source collection")
    for doi, leads in cases.items():
        _, body, _ = indexed[doi]
        # The title-free query text must differ from the indexed title and introduction.
        if leads["abstract_body"] in body or leads["abstract_conclusion"] in body:
            raise ValueError(f"Descriptive query copied from indexed text: {doi}")
    return dict(sorted(indexed.items())), ExcerptIndex(abstracts)


def role_result(sources, source, requirements, masked, method):
    if source is None or source == masked or source not in sources:
        return "unknown", []
    records = sources[source]
    if method == "same_author":
        matching = [i for i, roles in enumerate(records, 1) if requirements <= roles]
    elif method == "any_role":
        matching = [i for i, roles in enumerate(records, 1) if requirements & roles]
    else:
        # Document-level evidence cannot distinguish which author met both conditions.
        site_roles = set().union(*records)
        matching = list(range(1, len(records) + 1)) if requirements <= site_roles else []
    return ("supported" if matching else "no_matching_declaration"), matching


def evaluate_pair(declarations, candidate, doi, requirement, expected, masked, attribution):
    state, returned = role_result(declarations, candidate, requirement, doi if masked else None, attribution)
    truth = "unknown" if masked else ("supported" if expected else "no_matching_declaration")
    wanted = set() if masked else set(expected)
    actual = set(returned)
    unsupported = len(actual - wanted) if candidate == doi else len(actual)
    return {
        "source_correct": candidate == doi,
        "complete_success": candidate == doi and state == truth and actual == wanted,
        "any_unsupported_author": unsupported > 0,
        "unsupported_authors": unsupported,
        "state": state,
    }


def summarize_retrieval(rows):
    summary = {}
    for view in (*VIEWS, "mixed"):
        cohort = rows if view == "mixed" else [row for row in rows if row["view"] == view]
        controls = {}
        for method in METHODS:
            choices = [row["retrieval"][method] for row in cohort]
            controls[method] = {
                "top1": sum(item["selected"] == row["doi"] for item, row in zip(choices, cohort, strict=True)),
                "queries": len(cohort),
                "returned_source": sum(item["selected"] is not None for item in choices),
                "p50_p95_ms": list(map(float, np.quantile([np.median(item["ms"]) for item in choices], [.5, .95]))),
                "exact_title_routes": sum(item["title_route"] for item in choices),
                "exact_excerpt_routes": sum(item["excerpt_route"] for item in choices),
            }
        summary[view] = controls
    return summary


def summarize_attribution(rows):
    cohorts = {}
    for view in VIEWS:
        observed = [row for row in rows if row["view"] == view]
        for method in (*METHODS, "oracle_source"):
            for masked in (False, True):
                key = f"{view}/{method}/{'masked' if masked else 'visible'}"
                systems = {}
                for attribution in ATTRIBUTION:
                    records = [case[f"{method}/{attribution}/{masked}"]
                               for row in observed for case in row["pairs"]]
                    systems[attribution] = {
                        "checks": len(records),
                        "complete_success": sum(record["complete_success"] for record in records),
                        "queries_with_unsupported_author": sum(record["any_unsupported_author"] for record in records),
                        "unknown": sum(record["state"] == "unknown" for record in records),
                    }
                cohorts[key] = systems
    return cohorts


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cases, declarations, label_hash = controls(args.fixture, args.sources)
    indexed, excerpts = prepare_collection(args.collection, cases)
    identifiers = sorted(indexed)
    texts = [indexed[doi][1] for doi in identifiers]
    full_texts = [indexed[doi][2] for doi in identifiers]
    if len(set(texts)) != len(texts) or len(set(full_texts)) != len(full_texts):
        raise ValueError("Duplicate indexed texts")
    normalized = defaultdict(list)
    for doi, (title, _, _) in indexed.items():
        normalized[tuple(tokens(title))].append(doi)
    titles = {key: matches[0] for key, matches in normalized.items() if key and len(matches) == 1}

    checkpoint = json.loads((args.model / "checkpoint-provenance.json").read_bytes())
    for name, expected in checkpoint["files"].items():
        if digest(args.model / name) != expected:
            raise ValueError(f"Changed embedding file: {name}")
    model = QwenTorchEmbedding(args.model)
    model.model.config.use_cache = False
    bank = model.encode(texts)
    full_bank = model.encode(full_texts)
    model.encode([f"Instruct: {TASK}\nQuery: sample request"])
    with TemporaryDirectory() as folder:
        lexical_indexes = {}
        for method, index_texts in (("bm25", texts), ("bm25_full", full_texts)):
            directory = Path(folder) / method
            directory.mkdir()
            lexical_indexes[method] = build_lexical({text: i for i, text in enumerate(index_texts)}, directory)

    def search(lead, method, role_pair):
        start = perf_counter()
        uses_literal = method in ("literal_or_agreement", "literal_then_bm25_full")
        uses_title_lookup = uses_literal or method in ("exact_title_then_dense", "exact_title_or_agreement")
        title_match = titles.get(tuple(tokens(lead))) if uses_title_lookup else None
        excerpt_match = excerpts.unique_source(lead) if uses_literal and not title_match else None
        literal_source = title_match or excerpt_match
        needs_agreement = method in ("exact_title_or_agreement", "literal_or_agreement")
        lexical_method = "bm25_full" if method == "literal_then_bm25_full" else method
        lexical_selected = None
        if (lexical_method in lexical_indexes or needs_agreement) and not literal_source:
            lexical = lexical_indexes.get(lexical_method, lexical_indexes["bm25"]).document_scores(lead)
            if lexical:
                best = min(lexical, key=lambda hit: (-lexical[hit], identifiers[int(hit)]))
                lexical_selected = identifiers[int(best)]
        if literal_source:
            selected = literal_source
        elif lexical_method in lexical_indexes:
            selected = lexical_selected
        else:
            query = model.encode([f"Instruct: {TASK}\nQuery: {lead}"])[0]
            vectors = full_bank if method == "dense_full" else bank
            selected = identifiers[int(np.argmax(vectors @ query))]
            if needs_agreement and selected != lexical_selected:
                selected = None
        role_result(declarations, selected, role_pair, None, "same_author")
        return selected, (perf_counter() - start) * 1000, bool(title_match), bool(excerpt_match)

    for method in METHODS:
        first = next(iter(cases.values()))
        search(first["title"], method, frozenset(first["pairs"][0][:2]))

    rows = []
    for paper_index, (doi, leads) in enumerate(cases.items()):
        for view_index, view in enumerate(VIEWS):
            first_roles = frozenset(leads["pairs"][0][:2])
            outputs = defaultdict(list)
            for trial in range(REPEATS):
                offset = (paper_index + view_index + trial) % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    outputs[method].append(search(leads[view], method, first_roles))
            retrieval = {}
            selected = {"oracle_source": doi}
            for method, records in outputs.items():
                if len({result[0] for result in records}) != 1:
                    raise ValueError("Repeated source selection changed")
                selected[method] = records[0][0]
                retrieval[method] = {"selected": records[0][0], "title_route": records[0][2],
                                     "excerpt_route": records[0][3],
                                     "ms": [item[1] for item in records]}
            evaluations = []
            for role_a, role_b, authors in leads["pairs"]:
                row = {"roles": [role_a, role_b], "authors": authors}
                for method, source in selected.items():
                    for masked in (False, True):
                        for attribution in ATTRIBUTION:
                            row[f"{method}/{attribution}/{masked}"] = evaluate_pair(
                                declarations, source, doi, frozenset((role_a, role_b)), authors, masked, attribution)
                evaluations.append(row)
            rows.append(dict(doi=doi, view=view, retrieval=retrieval, pairs=evaluations))

    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "rows.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "source_papers": len(indexed), "judged_papers": len(cases), "repeats": REPEATS,
        "compound_cases": sum(len(paper["pairs"]) for paper in cases.values()),
        "compound_cases_with_match": sum(bool(authors) for paper in cases.values()
                                         for _, _, authors in paper["pairs"]),
        "retrieval": summarize_retrieval(rows),
        "attribution": summarize_attribution(rows),
        "gpu": model.torch.cuda.get_device_name(), "checkpoint": checkpoint,
        "queries_sha256": label_hash, "collection_sha256": COLLECTION_SHA256,
        "rows_sha256": digest(path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("benchmarks/plos-credit-pilot-v1"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
