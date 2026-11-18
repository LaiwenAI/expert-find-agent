"""Retrieve an article, split a plain-text conjunction, and return source-bound authors."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.attribution_controls import LABELS, Reader
from experiments.credit.conjunctive_source_retrieval import (
    TASK as SOURCE_TASK, edit_conclusion, prepare_collection, sentences,
)
from experiments.credit.counterfactual_role_edits import POSITIVE, fingerprint, label
from experiments.credit.evaluate_role_edits import MEANINGS, prompt
from experiments.credit.factored_role_edits import candidate_lists, choose_batched
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_language_router import NISO_SHA256, TASK, Router
from experiments.credit.role_prototypes import evaluate_atomic, train_banks
from experiments.qasper.adaptive_evidence import digest
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("bm25_direct", "bm25_factored", "dense_joint", "oracle_boundaries")
VIEWS = ("title", "edited_conclusion")
FORMATS = (
    ('In the study described as "{lead}", find authors who did both {activity}.',
     re.compile(r'In the study described as "(?P<lead>[^"]+)", find authors who did both (?P<activity>.+)\.')),
    ('Find authors responsible for both {activity} in the study described as "{lead}".',
     re.compile(r'Find authors responsible for both (?P<activity>.+) in the study described as "(?P<lead>[^"]+)"\.')),
)
CONNECTIVES = re.compile(r"\s+(?:and also|as well as|and)\s+", re.IGNORECASE)


def spans(activity):
    return [(activity[:m.start()].strip(), activity[m.end():].strip()) for m in CONNECTIVES.finditer(activity)
            if len(activity[:m.start()].split()) >= 2 and len(activity[m.end():].split()) >= 2]


def query(task, value):
    return f"Instruct: {task}\nQuery: {value}"


def split_request(text):
    for _, pattern in FORMATS:
        matched = pattern.fullmatch(text)
        if matched:
            return matched["lead"], matched["activity"]
    raise ValueError("Unrecognized source and activity envelope")


class Endpoint:
    def __init__(self, router, reader, index, ids, documents, source_roles):
        self.router = router
        self.reader = reader
        self.index = index
        self.ids = ids
        self.source_roles = source_roles
        self.source_bank = router.embedder.encode(documents)

    def bm25(self, lead):
        hits = self.index.document_scores(lead)
        return (self.ids[int(min(hits, key=lambda i: (-hits[i], self.ids[int(i)])))] if hits else None)

    @torch.inference_mode()
    def run(self, text, method, rotation, option_order, oracle=None, *, proposal_bank=None, read_roles=True):
        started = perf_counter()
        lead, activity = split_request(text)
        selected = self.bm25(lead) if method != "dense_joint" else None
        selected_atoms = None
        predicted_roles = None
        rendered = []
        segments = []
        if method != "bm25_direct":
            segments = [oracle] if method == "oracle_boundaries" else spans(activity)
            atoms = list(dict.fromkeys(atom for pair in segments for atom in pair))
            rendered = [query(TASK, atom) for atom in atoms]
            if method == "dense_joint":
                rendered.append(query(SOURCE_TASK, lead))
        vector_batch = self.router.embedder.encode(rendered) if rendered else None
        if method == "dense_joint":
            selected = self.ids[int(np.argmax(self.source_bank @ vector_batch[-1]))]
        records = self.source_roles.get(selected)
        returned = []
        calls = 0
        complete = False
        if records and method == "bm25_direct":
            prompts = [prompt("Which authors did both " + activity + "?", author, [(author, source)], option_order)
                       for author, source in enumerate(records, 1)]
            choices, _ = self.reader.classify(prompts)
            decoded = [option_order[LABELS.index(symbol)] for symbol, _ in choices]
            returned = [a for a, result in enumerate(decoded, 1) if result == POSITIVE]
            complete = "unavailable" not in decoded
            calls = 1
        elif records and segments:
            matrix = vector_batch[:len(atoms)] @ self.router.vectors.T
            values = dict(zip(atoms, matrix, strict=True))
            # Score each cut by the best product-space role pair with two distinct roles.
            def split_score(pair):
                scores = values[pair[0]][:, None] + values[pair[1]][None, :]
                np.fill_diagonal(scores, -np.inf)
                return float(scores.max())
            selected_atoms = max(segments, key=split_score)
            if proposal_bank is None:
                proposal_scores = [values[atom] for atom in selected_atoms]
            else:
                positions = {atom: i for i, atom in enumerate(atoms)}
                proposal_scores = vector_batch[[positions[atom] for atom in selected_atoms]] @ proposal_bank.T
            dense_ranks = [np.argsort(-scores, kind="stable").tolist() for scores in proposal_scores]
            if read_roles:
                shortlist = candidate_lists(self.router, selected_atoms, dense_ranks, 3)
                predictions = choose_batched(self.router, selected_atoms, shortlist, rotation)
                calls = 1
            else:
                predictions = [rank[0] for rank in dense_ranks]
            predicted_roles = [ROLES[i] for i in predictions]
            if len(set(predicted_roles)) == 2:
                returned = [a for a, source in enumerate(records, 1) if label(source, predicted_roles) == POSITIVE]
                complete = True
        citations = [[selected, author] for author in returned]
        return dict(source=selected, complete=complete, returned=returned, citations=citations,
                    roles=predicted_roles, calls=calls, cuts=len(segments),
                    boundary_exact=selected_atoms == oracle if selected_atoms is not None else False,
                    ms=(perf_counter()-started)*1000)


def summarize(rows, methods=METHODS, views=VIEWS, reference_name="bm25_direct"):
    metrics = {}
    for lead_style in views:
        group = [row for row in rows if row["lead"] == lead_style]
        papers = [row["doi"] for row in group]
        report = {}
        for method in methods:
            runs = [row["arms"][method] for row in group]
            reference = [row["arms"][reference_name] for row in group]
            report[method] = dict(
                requests=len(runs), source_exact=sum(item["source"] == row["doi"] for item, row in zip(runs, group)),
                multi_cut=sum(item["cuts"] > 1 for item in runs),
                boundary_exact=sum(item["boundary_exact"] for item in runs),
                roles_exact=sum(item["roles_exact"] for item in runs),
                full_author_set_exact=sum(item["success"] for item in runs),
                unresolved=sum(not item["complete"] for item in runs),
                unsupported_promotions=sum(item["unsupported"] for item in runs),
                p50_p95_ms=np.quantile([item["ms"] for item in runs], [.5, .95]).tolist(),
                success_delta=clustered_paired_interval(
                    [int(run["success"]) - int(ref["success"]) for run, ref in zip(runs, reference)], papers),
            )
        metrics[lead_style] = report
    return metrics


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.glossary) != NISO_SHA256:
        raise ValueError("Changed NISO snapshot")
    upstream = json.loads(args.glossary.read_bytes())
    glossary = [f"{role}: {upstream['definitions'][role]}" for role in ROLES]
    phrases = defaultdict(list)
    for example in upstream["examples"]:
        phrases[example["role"]].append(example["text"])
    manifest = json.loads((args.fixture / "manifest.json").read_bytes())
    path = args.fixture / "pairs.jsonl"
    if digest(path) != manifest["pairs_sha256"]:
        raise ValueError("Changed counterfactual fixture")
    groups = [g for line in path.read_text().splitlines() if (g := json.loads(line))["split"] == "test"]
    source_roles = {r["doi"]: r["roles_by_author"] for r in manifest["sources"]}
    collection, _ = prepare_collection(args.collection, {})
    doi_list = sorted(collection)
    documents = [collection[doi][2] for doi in doi_list]
    if not source_roles.keys() <= collection.keys() or len(set(documents)) != len(documents):
        raise ValueError("Mismatched collection")
    embedding = json.loads((args.embedding / "checkpoint-provenance.json").read_bytes())
    for name, expected in embedding["files"].items():
        if digest(args.embedding / name) != expected:
            raise ValueError(f"Changed embedding checkpoint: {name}")
    embedder = QwenTorchEmbedding(args.embedding)
    embedder.model.config.use_cache = False
    tokenizer, reader_model, checkpoint = load_reader(args.reader)
    reader = Reader(tokenizer, reader_model)
    with TemporaryDirectory() as root:
        directory = Path(root)
        (directory / "glossary").mkdir()
        (directory / "sources").mkdir()
        lexical = build_lexical({text: i for i, text in enumerate(glossary)}, directory / "glossary")
        index = build_lexical({text: i for i, text in enumerate(documents)}, directory / "sources")
    router = Router(glossary, lexical, embedder.encode(glossary), embedder, reader_model, tokenizer, 3)
    endpoint = Endpoint(router, reader, index, doi_list, documents, source_roles)
    variants = {method: (method, {}) for method in METHODS}
    views, reference = VIEWS, "bm25_direct"
    fit = None
    if args.role_training:
        banks, fit = train_banks(args.role_training, router, upstream["examples"])
        fit["input_sha256"] = digest(args.role_training)
        fit["atomic"] = evaluate_atomic(router, banks, upstream["examples"])
        variants = {
            f"{name}_{'reader' if use_reader else 'embed'}":
                ("bm25_factored", dict(proposal_bank=bank if name != "definitions" else None, read_roles=use_reader))
            for name, bank in banks.items() for use_reader in (False, True)
        }
        views, reference = ("edited_conclusion",), "definitions_reader"
    methods = tuple(variants)
    rows = []
    for i, group in enumerate(groups):
        doi = group["doi"]
        title, _, full_text = collection[doi]
        description = edit_conclusion(sentences(full_text.split("\n", 1)[1])[-1])
        rng = int(fingerprint(group["id"], "source")[:8], 16)
        template, _ = FORMATS[rng % len(FORMATS)]
        connector = (" and ", " as well as ")[(rng // len(FORMATS)) % 2]
        rotation = rng // (2 * len(FORMATS))
        option_order = sorted(MEANINGS, key=lambda meaning: fingerprint(group["id"], meaning))
        for lead_style, value in zip(VIEWS, (title, description), strict=True):
            if lead_style not in views:
                continue
            for view_index, view in enumerate(("before", "after")):
                roles = group[view]
                pieces = [phrases[r][int(fingerprint(group["id"], r), 16) % len(phrases[r])] for r in roles]
                oracle = tuple(" ".join(piece.split()).rstrip(".?!") for piece in pieces)
                text = template.format(lead=value.replace('"', "'"), activity=connector.join(oracle))
                wanted = [j for j, evidence in enumerate(source_roles[doi], 1) if label(evidence, roles) == POSITIVE]
                wanted_pairs = {(doi, j) for j in wanted}
                shift = (i + view_index) % len(methods)
                if not rows:
                    for method, settings in variants.values():
                        endpoint.run(text, method, rotation, option_order, oracle, **settings)
                arms = {}
                for variant in methods[shift:] + methods[:shift]:
                    method, settings = variants[variant]
                    item = endpoint.run(text, method, rotation, option_order, oracle, **settings)
                    citations = set(map(tuple, item["citations"]))
                    item["unsupported"] = len(citations - wanted_pairs)
                    item["roles_exact"] = item["roles"] is not None and set(item["roles"]) == set(roles)
                    item["success"] = item["complete"] and item["source"] == doi and citations == wanted_pairs
                    arms[variant] = item
                rows.append(dict(id=group["id"], doi=doi, view=view, lead=lead_style, wanted=wanted, arms=arms))
    with args.output.open("x") as stream:
        json.dump(dict(summary=summarize(rows, methods, views, reference), rows=rows, collection_sha256=digest(args.collection),
                       fixture_sha256=digest(path), script_sha256=digest(Path(__file__)), embedding=embedding,
                       checkpoint=checkpoint, gpu=torch.cuda.get_device_name(), fit=fit), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "collection", "glossary", "embedding", "reader", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--role-training", type=Path)
    run(parser.parse_args())
