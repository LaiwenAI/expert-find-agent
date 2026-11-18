"""Test whether contributor links help retrieve two specified publications."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.conjunctive_source_retrieval import TASK as SOURCE_TASK, sentences
from experiments.credit.counterfactual_role_edits import fingerprint
from experiments.credit.cross_source_identity import eligible, load, pairs, xml_sources
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, TASK
from experiments.credit.role_prototypes import normalized, train_banks
from experiments.qasper.adaptive_evidence import digest
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("bm25", "dense", "fused", "identity_first", "corroborated",
           "bm25_identity", "bm25_one_side", "oracle_roles")
LEXICAL_ONLY = {"bm25", "bm25_identity", "bm25_one_side"}
STYLES = ("title", "abstract", "title_sketch")
FORMAT = re.compile(
    r'Across "([^"]+)" and "([^"]+)", find the same person who did (.+?) in the first study'
    r' and (.+) in the second study\.\Z',
)
K = 4


def searchable_text(folders):
    documents = {}
    for folder in folders:
        for file in xml_sources(folder):
            meta = ET.fromstring(file.read_bytes()).find("./front/article-meta")
            doi = meta.findtext("article-id[@pub-id-type='doi']")
            title = " ".join("".join(meta.find("title-group/article-title").itertext()).split())
            body = " ".join("".join(p.itertext()) for p in meta.findall("./abstract//p"))
            documents[doi] = title, " ".join(body.split())
    return documents


def lead(document, style):
    title, abstract = document
    if style == "title":
        return title
    if style == "abstract":
        parsed = sentences(abstract)
        return parsed[len(parsed) // 2] if parsed else title
    terms = normalized(title).split()
    if len(terms) < 5:
        return title
    first = len(terms) // 3
    return " ".join(terms[first:first + 4])


class Glossary:
    def __init__(self, embedder, glossary):
        self.embedder = embedder
        self.vectors = embedder.encode(glossary)


class Search:
    def __init__(self, encoder, index, documents, evidence, ridge):
        self.ids = sorted(documents)
        self.encoder = encoder
        self.index = index
        self.evidence = evidence
        self.ridge = ridge
        self.bank = encoder.encode(["\n".join(documents[doi]) for doi in self.ids])

    def lexical(self, text):
        scores = self.index.document_scores(text)
        return [self.ids[int(i)] for i in sorted(scores, key=lambda i: (-scores[i], self.ids[int(i)]))[:K]]

    def identify(self, first, second, roles):
        if first == second or first not in self.evidence or second not in self.evidence or roles[0] == roles[1]:
            return [], False
        a = eligible(self.evidence[first], roles[0])
        b = eligible(self.evidence[second], roles[1])
        matches = pairs(a, b, "person")
        complete = all(author["roles"] for doi in (first, second) for author in self.evidence[doi]["authors"])
        return matches, bool(matches) or (complete and all(x["person"] for x in a + b))

    def run(self, request, method, reference_roles=None):
        start = perf_counter()
        parsed = FORMAT.fullmatch(request)
        if not parsed:
            raise ValueError("Unexpected two-publication request")
        left, right, first, second = parsed.groups()
        inputs = [f"Instruct: {TASK}\nQuery: {value}" for value in (first, second)]
        if method not in LEXICAL_ONLY:
            inputs.extend(f"Instruct: {SOURCE_TASK}\nQuery: {value}" for value in (left, right))
        vectors = self.encoder.encode(inputs)
        roles = [ROLES[i] for i in np.argmax(vectors[:2] @ self.ridge.T, axis=1).tolist()]
        if method == "oracle_roles":
            roles = reference_roles
        dense, lexical = [], []
        if method not in LEXICAL_ONLY:
            order = np.argsort(-(vectors[2:] @ self.bank.T), axis=1, kind="stable")
            dense = [[self.ids[i] for i in ranked[:K]] for ranked in order]
        if method != "dense":
            lexical = [self.lexical(value) for value in (left, right)]
        if method in LEXICAL_ONLY:
            banks = lexical
        elif method == "dense":
            banks = dense
        else:
            banks = []
            for sparse, semantic in zip(lexical, dense, strict=True):
                scores = {}
                for rank in (sparse, semantic):
                    for i, doi in enumerate(rank, 1):
                        scores[doi] = scores.get(doi, 0) + 1 / (60 + i)
                banks.append(sorted(scores, key=lambda x: (-scores[x], x))[:K])
        source = tuple(r[0] if r else None for r in banks)
        initial = source
        links, resolved = self.identify(*source, roles)
        if method in ("identity_first", "corroborated", "bm25_identity", "bm25_one_side") and not links:
            best = None
            for i, a in enumerate(banks[0]):
                for j, b in enumerate(banks[1]):
                    if method == "bm25_one_side" and (int(a != initial[0]) + int(b != initial[1])) != 1:
                        continue
                    if method == "corroborated" and any(
                        doi != current and (doi not in bm[:3] or doi not in de[:3]
                                            or (bm and bm[0] == de[0]))
                        for doi, current, bm, de in zip((a, b), initial, lexical, dense, strict=True)
                    ):
                        continue
                    verified, complete = self.identify(a, b, roles)
                    if verified and (best is None or (i + j, i, j) < best):
                        best = i + j, i, j
                        source, links, resolved = (a, b), verified, complete
        citations = [[source[0], i, source[1], j] for i, j in links]
        return dict(source=source, initial=initial, roles=roles, resolved=resolved, citations=citations,
                    banks=banks, duration=(perf_counter() - start) * 1000)


def summarize(rows):
    result = {}
    for style in STYLES:
        result[style] = {}
        current = [row for row in rows if row["style"] == style]
        for method in METHODS:
            runs = [row["arms"][method] for row in current]
            result[style][method] = dict(
                requests=len(runs), sources_exact=sum(r["sources_exact"] for r in runs),
                roles_exact=sum(r["roles_exact"] for r in runs), exact=sum(r["exact"] for r in runs),
                positive_exact=sum(r["exact"] for row, r in zip(current, runs) if row["kind"] == "positive"),
                added_wrong_source_links=sum(r["unsupported"] for r in runs if not r["sources_exact"]),
                unsupported=sum(r["unsupported"] for r in runs),
                source_switches=sum(r["source"] != r["initial"] for r in runs),
                no_link_promotions=sum(bool(r["citations"]) for row, r in zip(current, runs) if row["kind"] == "no_link"),
                p50_p95_ms=np.quantile([r["duration"] for r in runs], [.5,.95]).tolist(),
            )
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.fixture / "manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        if digest(args.fixture / name) != expected:
            raise ValueError(f"Changed fixture: {name}")
    queries = json.loads((args.fixture / "queries.json").read_bytes())
    documents = searchable_text(args.sources)
    ownership = load(args.sources)
    expected = {source["doi"]: source["source_sha256"] for source in manifest["input_sources"]}
    if {doi: s["source_sha256"] for doi, s in ownership.items()} != expected or ownership.keys() != documents.keys():
        raise ValueError("Changed XML inputs")
    if digest(args.glossary) != NISO_SHA256:
        raise ValueError("Changed role glossary")
    snapshot = json.loads(args.glossary.read_bytes())
    paraphrases = json.loads(args.phrases.read_bytes())
    provenance = json.loads((args.embedding / "checkpoint-provenance.json").read_bytes())
    for name, expected_hash in provenance["files"].items():
        if digest(args.embedding / name) != expected_hash:
            raise ValueError(f"Changed embedding file: {name}")
    embedder = QwenTorchEmbedding(args.embedding)
    embedder.model.config.use_cache = False
    bank = Glossary(embedder, [f"{role}: {snapshot['definitions'][role]}" for role in ROLES])
    banks, fit = train_banks(args.training, bank, snapshot["examples"])
    with TemporaryDirectory() as scratch:
        texts = ["\n".join(documents[doi]) for doi in sorted(documents)]
        if len(set(texts)) != len(texts):
            raise ValueError("Duplicate searchable documents")
        lexical = build_lexical({text: i for i, text in enumerate(texts)}, Path(scratch))
    search = Search(embedder, lexical, documents, ownership, banks["ridge"])
    rows = []
    for i, q in enumerate(queries):
        source = q["left"], q["right"]
        roles = [q["left_role"], q["right_role"]]
        gold = {(source[0], a, source[1], b) for a, b in q["gold"]}
        text = [paraphrases[r][int(fingerprint(q["id"], r), 16) % len(paraphrases[r])] for r in roles]
        for style in STYLES:
            left, right = [lead(documents[doi], style).replace('"', "'") for doi in source]
            request = (f'Across "{left}" and "{right}", find the same person who did {text[0]} in the first study'
                       f' and {text[1]} in the second study.')
            if not rows:
                for method in METHODS:
                    search.run(request, method, roles)
            estimates = {}
            offset = i % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                result = search.run(request, method, roles)
                found = set(map(tuple, result["citations"]))
                result.update(
                    sources_exact=result["source"] == source,
                    roles_exact=result["roles"] == roles,
                    exact=result["source"] == source and result["resolved"] and found == gold,
                    unsupported=len(found - gold),
                )
                estimates[method] = result
            rows.append(dict(id=q["id"], left=source[0], right=source[1], kind=q["kind"], style=style, arms=estimates))
    with args.output.open("x") as stream:
        json.dump(dict(summary=summarize(rows), rows=rows, fit=fit, embedding=provenance,
                       fixture_sha256=digest(args.fixture / "manifest.json"), script_sha256=digest(Path(__file__))),
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "glossary", "training", "phrases", "embedding", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    run(parser.parse_args())
