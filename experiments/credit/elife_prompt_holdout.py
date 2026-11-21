"""Freeze a disjoint eLife cohort for a fresh role prompt comparison."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

from experiments.credit.elife_role_nomination import REVISION, UPSTREAM, parse, projection
from experiments.credit.engineering_cross_source_panel import SENTENCE, requests
from experiments.credit.topic_role_holdout import identities, metadata, title
from experiments.credit.topic_role_postings import digest, write_json


def inventory(folders):
    result = defaultdict(set)
    prior = set()
    for folder in folders:
        for path in sorted(folder.rglob("*.xml")):
            raw = path.read_bytes()
            checksum = sha256(raw).hexdigest()
            if checksum in prior:
                continue
            prior.add(checksum)
            meta = metadata(raw)
            if meta is None:
                raise ValueError(f"Missing publisher metadata: {path.name}")
            ids, names = identities(meta)
            result["orcid"].update(ids)
            result["name"].update(names)
            result["title"].add(title(meta).casefold())
            result["doi"].update(node.text.strip().casefold() for node in meta.findall("article-id[@pub-id-type='doi']")
                                 if node.text)
    if not prior:
        raise ValueError("Require the prior publisher XML inventory")
    return result, sorted(prior)


def build(projection_path, manifest):
    if manifest["revision"] != REVISION or manifest["projection_sha256"] != digest(projection_path):
        raise ValueError("Unexpected holdout source snapshot")
    articles, people = {}, defaultdict(set)
    counts = Counter()
    for row in json.loads(projection_path.read_bytes())["sources"]:
        doi = row["doi"]
        header, abstract = row["text"].split("\n", 1)
        sentences = SENTENCE.split(abstract.replace("\n", " "))
        roles = {author["person"] or f"{doi}#{author['ordinal']}": author["roles"] for author in row["authors"]}
        if not row["declaration_complete"] or len(sentences) < 3 or len(roles) != len(row["authors"]):
            counts["incomplete_or_ambiguous"] += 1
            continue
        middle = len(sentences) // 2
        articles[doi] = dict(indexed=header + "\n" + " ".join(sentences[:middle]),
                             clue=" ".join(sentences[middle:]), roles=roles)
        for author in row["authors"]:
            if author["person"]:
                people[author["person"]].add(doi)
    links = [(pair, person) for person, dois in people.items() for pair in combinations(sorted(dois), 2)]
    rows = {(row["family"], row["variant"]): row for row in requests(articles, links)}
    if not rows:
        raise ValueError("No independent holdout families")
    cases = [rows[key] for key in sorted(rows)]
    paper_pairs = {tuple(row["papers"]) for row in cases}
    linked_papers = set().union(*map(set, paper_pairs))
    return articles, cases, dict(admitted=len(articles), dropped=dict(counts), links=len(links),
                                 paper_pairs=len(paper_pairs), linked_papers=len(linked_papers),
                                 contributors=len({key for row in cases for key in row["gold"]}),
                                 variants=dict(Counter(row["variant"] for row in cases)))


def sample_families(rows):
    """Keep one complete query family for each unique target paper pair."""
    first = {}
    for row in rows:
        pair = tuple(row["papers"])
        first[pair] = min(first.get(pair, row["family"]), row["family"])
    selected = [row for row in rows if first[tuple(row["papers"])] == row["family"]]
    by_pair = defaultdict(list)
    for row in selected:
        by_pair[tuple(row["papers"])].append(bool(row["gold"]))
    if any(sorted(labels) != [False, True] for labels in by_pair.values()):
        raise ValueError("Expected a complete positive and control family for every pair")
    return selected


def freeze(sample, xml, reference_manifest, projection_path, output, prior_folders):
    if output.exists() or projection_path.exists():
        raise ValueError("Require fresh outputs")
    files = sample.read_text().splitlines()
    if len(files) != 512 or len(set(files)) != len(files):
        raise ValueError("Expected 512 distinct pinned source names")
    old = json.loads(reference_manifest.read_bytes())
    if old["revision"] != REVISION:
        raise ValueError("Unexpected prior publisher revision")
    old_ids = {str(int(s["filename"].split("-")[1])) for s in old["source_inventory"]}
    prior, hashes = inventory(prior_folders)
    records, kept = [], []
    for name in files:
        match = re.fullmatch(r"elife-(\d+)-v\d+\.xml", name)
        if not match or not 65000 <= int(match[1]) < 84000 or match[1] in old_ids:
            raise ValueError(f"Out of holdout sample scope: {name}")
        raw = (xml / name).read_bytes()
        row, excluded = parse(raw, name)
        if row is not None:
            if row["doi"].casefold() in prior["doi"] or row["title"].casefold() in prior["title"]:
                excluded = "prior_doi_or_title"
            elif row["author_ids"] & prior["orcid"] or row["author_names"] & prior["name"]:
                excluded = "prior_author"
        records.append(dict(filename=name, sha256=sha256(raw).hexdigest(), exclusion=excluded))
        if excluded is None:
            kept.append(row)
    write_json(projection_path, projection(kept))
    manifest = dict(upstream=UPSTREAM, revision=REVISION, source_inventory=records,
                    prior_manifest_sha256=digest(reference_manifest), prior_documents=len(hashes),
                    prior_inventory_sha256=sha256("\n".join(hashes).encode()).hexdigest(),
                    projection_sha256=digest(projection_path))
    _, _, census = build(projection_path, manifest)
    write_json(output, manifest | {"panel": census})
    print(json.dumps(census))


def run(projection_path, manifest_path, niso, training, bands, frozen, embedding, reader, output, reverse=False):
    import os
    from copy import copy

    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.compact_role_prompts import COMPACT, evaluate
    from experiments.credit.engineering_one_role_recheck import RoleReader
    from experiments.credit.engineering_role_banks import ROLES, projections, queries
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, cases, census = build(projection_path, json.loads(manifest_path.read_bytes()))
    definitions, training_text, wording = queries(niso, training)
    checkpoint = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training_text)
    full = RoleReader(reader, definitions)
    compact, names = copy(full), copy(full)
    compact.definitions = [f"{role}: {COMPACT[role]}" for role in ROLES]
    names.definitions = ROLES
    models = {"full": full, "compact": compact, "name_only": names}
    role_bands = json.loads(bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    stats = evaluate(articles, cases, encoder, banks, wording, models, role_bands, cutoff, reverse)
    write_json(output, dict(
        manifest_sha256=digest(manifest_path), projection_sha256=digest(projection_path),
        script_sha256=digest(Path(__file__)), panel=census, embedding_revision=checkpoint["revision"],
        reader_revision=full.provenance["revision"], bands=role_bands, cutoff=cutoff,
        gpu=torch.cuda.get_device_name(), reverse=reverse, results=stats,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    parser = commands.add_parser("freeze")
    for argument in ("sample", "xml", "reference_manifest", "projection_path", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--prior-folders", nargs="+", type=Path, required=True)
    parser = commands.add_parser("run")
    for argument in ("projection_path", "manifest_path", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = vars(cli.parse_args())
    {"freeze": freeze, "run": run}[args.pop("command")](**args)
