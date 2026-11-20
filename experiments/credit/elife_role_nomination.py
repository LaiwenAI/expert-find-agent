"""Freeze and evaluate an independent publisher-declared expert search catalog."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.credit.plos_role_inventory import ROLES


UPSTREAM = "https://github.com/elifesciences/elife-article-xml"
REVISION = "58250dd64e27afae6bc2700bcddd84dd38776e0d"
LICENSES = {"https://creativecommons.org/licenses/by/4.0/",
            "https://creativecommons.org/publicdomain/zero/1.0/"}
ARMS = ("topic_four", "topic_eight", "person_four", "bounded_four", "union_four")


def normalized_role(label):
    label = re.sub(r"\s*[-–—]\s*", " – ", pilot.compact(label)).casefold()
    return label.replace(" & ", " and ")


CANONICAL = {normalized_role(name): name for name in ROLES}


def parse(raw, filename):
    from xml.etree import ElementTree as ET

    root = ET.fromstring(raw)
    if root.get("article-type") != "research-article":
        return None, "article_type"
    meta = root.find("./front/article-meta")
    doi = meta.findtext("article-id[@pub-id-type='doi']", "")
    article_id = re.fullmatch(r"elife-(\d+)-v\d+\.xml", filename)[1]
    if doi.casefold() != f"10.7554/elife.{article_id}":
        return None, "article_identity"
    license_node = meta.find("permissions/license")
    href = "" if license_node is None else license_node.get("{http://www.w3.org/1999/xlink}href", "")
    license_url = href.replace("http://", "https://", 1)
    if license_url not in LICENSES:
        return None, "license"
    title = holdout.title(meta)
    abstract = next(
        (node for node in meta.findall("abstract") if "abstract-type" not in node.attrib), None)
    text = "" if abstract is None else pilot.compact(" ".join(abstract.itertext()))
    if not title or not text:
        return None, "missing_text"
    heading = {pilot.compact(" ".join(node.itertext()))
               for node in meta.findall("article-categories/subj-group[@subj-group-type='heading']/subject")}
    keyword = {pilot.compact(" ".join(node.itertext())).casefold()
               for node in meta.findall("kwd-group[@kwd-group-type='author-keywords']/kwd")}
    notes = {note.get("id"): "".join(note.itertext())
             for note in root.findall("./back//fn[@fn-type='con']") if note.get("id")}
    authors = []
    fully_declared = True
    for ordinal, author in enumerate(meta.findall("contrib-group/contrib[@contrib-type='author']"), 1):
        direct = ["".join(role.itertext()) for role in author.findall("role")]
        ids = {rid for link in author.findall("xref[@ref-type='fn']")
               for rid in link.get("rid", "").split()}
        entries = direct or [part for rid in sorted(ids & notes.keys()) for part in notes[rid].split(",")]
        labels = [CANONICAL.get(normalized_role(entry)) for entry in entries if entry.strip()]
        roles = sorted({role for role in labels if role is not None})
        fully_declared &= bool(roles) and None not in labels
        identifiers = {valid for item in author.findall("contrib-id[@contrib-id-type='orcid']")
                       if (valid := holdout.orcid_id(item.text or ""))}
        identity = sha256(f"elife-role-search:{min(identifiers)}".encode()).hexdigest()[:24] if len(identifiers) == 1 else None
        authors.append(dict(ordinal=ordinal, person=identity, roles=roles))
    if not authors or not heading:
        return None, "missing_authors_or_heading"
    ids, names = holdout.identities(meta)
    return dict(
        doi=doi, text=f"{title}\n{text}", topics=sorted(heading - {""}),
        keywords=sorted(keyword - {""}), authors=authors, license=license_url,
        declaration_complete=fully_declared,
        title=title, author_ids=ids, author_names=names,
    ), None


def projection(rows):
    return {"sources": [
        {k: v for k, v in row.items() if k not in ("title", "author_ids", "author_names")}
        for row in sorted(rows, key=lambda row: row["doi"])
    ]}


def freeze(sample, xml_dir, manifest, output, prior_folders):
    if manifest.exists() or output.exists():
        raise ValueError("Provide fresh outputs")
    prior_ids, prior_names, prior_titles, prior_digests = set(), set(), set(), set()
    for directory in prior_folders:
        for path in sorted(directory.glob("*.xml")):
            raw = path.read_bytes()
            prior_digests.add(sha256(raw).hexdigest())
            meta = holdout.metadata(raw)
            ids, names = holdout.identities(meta)
            prior_ids.update(ids)
            prior_names.update(names)
            prior_titles.add(holdout.title(meta).casefold())
    if not prior_digests:
        raise ValueError("Prior PLOS source inventory is empty")
    inventory, rows = [], []
    files = sample.read_text().splitlines()
    if len(files) != 360 or len(set(files)) != len(files):
        raise ValueError("Expected the frozen 360-file sample")
    for name in files:
        raw = (xml_dir / name).read_bytes()
        row, reason = parse(raw, name)
        if row and (row["title"].casefold() in prior_titles or row["author_ids"] & prior_ids
                    or row["author_names"] & prior_names):
            reason = "prior_title_or_author"
        inventory.append(dict(filename=name, sha256=sha256(raw).hexdigest(), exclusion=reason))
        if reason is None:
            rows.append(row)
    pilot.write_json(output, projection(rows))
    pilot.write_json(manifest, dict(
        upstream=UPSTREAM, revision=REVISION, source_inventory=inventory,
        prior_documents=len(prior_digests), prior_inventory_sha256=sha256(
            "\n".join(sorted(prior_digests)).encode()).hexdigest(),
        projection_sha256=pilot.digest(output),
    ))


def prepare(xml_dir, manifest, output):
    frozen = json.loads(manifest.read_bytes())
    if output.exists() or frozen["revision"] != REVISION:
        raise ValueError("Expected a fresh output and the pinned revision")
    rows = []
    for entry in frozen["source_inventory"]:
        if entry["exclusion"] is None:
            raw = (xml_dir / entry["filename"]).read_bytes()
            if sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Snapshot differs: {entry['filename']}")
            row, reason = parse(raw, entry["filename"])
            if reason is not None:
                raise ValueError(f"Invalid included source: {entry['filename']}")
            rows.append(row)
    pilot.write_json(output, projection(rows))
    if pilot.digest(output) != frozen["projection_sha256"]:
        raise ValueError("Rebuilt projection differs")


def evaluate_panel(catalog, topics, witnesses=None):
    from time import perf_counter

    queries = list(pilot.query_panel(catalog, topics))
    if not queries:
        return {"queries": 0}
    results = {pct: {arm: [] for arm in ARMS} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        start = perf_counter()
        ranked = catalog.rank(topic)
        rank_ms = (perf_counter() - start) * 1000
        shift = index % len(ARMS)
        covers = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for pct in covers:
            for arm in ARMS[shift:] + ARMS[:shift]:
                measured = nomination.measure(catalog, topic, pair, ranked, pct, arm,
                                              (holdout.CUTOFF,), rank_ms)
                results[pct][arm].append(measured[holdout.CUTOFF])
    if witnesses is not None:
        witnesses.update({
            pct: {arm: [
                dict(topic=row["topic"], citations=[catalog.articles[source]["doi"]
                                                   for _, source in row["witnesses"]])
                for row in rows
            ] for arm, rows in estimates.items()}
            for pct, estimates in results.items()
        })
    return dict(
        queries=len(queries), topics=len({topic for topic, _, _ in queries}),
        masks={pct: {arm: {base: holdout.analyze(rows, estimates[base])
                           for base in ("topic_four", "topic_eight", "union_four")}
                     for arm, rows in estimates.items()} for pct, estimates in results.items()},
    )


def run(projection, manifest, embedding, reranker, reference, output, witness_output=None):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if (output.exists() or (witness_output is not None and witness_output.exists())
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if pilot.digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Expected the pinned eLife projection")
    old = json.loads(reference.read_bytes())
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (old["embedding"], old["reranker"]):
        raise ValueError("Expected the original PLOS checkpoints")
    articles = json.loads(projection.read_bytes())["sources"]
    catalog = pilot.Catalog(articles, encoder, reader)
    complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
    catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
    headings = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= 8)
    catalog.verify(headings[0], catalog.rank(headings[0])[:8])
    witnesses = dict(headings={}, keywords={}) if witness_output is not None else None
    heading = evaluate_panel(catalog, headings, None if witnesses is None else witnesses["headings"])
    catalog.topics.clear()
    for index, article in enumerate(articles):
        for topic in article["keywords"]:
            catalog.topics[topic].add(index)
    keywords = sorted(t for t, ids in catalog.topics.items() if len(ids) >= 4)
    keyword = evaluate_panel(catalog, keywords, None if witnesses is None else witnesses["keywords"])
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), manifest_sha256=pilot.digest(manifest),
        reference_sha256=pilot.digest(reference), embedding=encoder_info["revision"],
        reranker=reader.provenance["revision"], cutoff=holdout.CUTOFF,
        retained_articles=len(articles), indexed={pct: len(ids) for pct, ids in catalog.visible.items()},
        exclusions=dict(Counter(row["exclusion"] for row in frozen["source_inventory"] if row["exclusion"])),
        headings=heading, keywords=keyword,
    ))
    if witnesses is not None:
        pilot.write_json(witness_output, dict(
            projection_sha256=pilot.digest(projection), replay_sha256=pilot.digest(output),
            witnesses=witnesses,
        ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    commands = {"freeze": freeze, "prepare": prepare, "run": run}
    for command, arguments in {
        "freeze": ("sample", "xml_dir", "manifest", "output"),
        "prepare": ("xml_dir", "manifest", "output"),
        "run": ("projection", "manifest", "embedding", "reranker", "reference", "output"),
    }.items():
        parser = sub.add_parser(command)
        for name in arguments:
            parser.add_argument(name, type=Path)
        if command == "freeze":
            parser.add_argument("--prior-folders", nargs="+", type=Path, required=True)
        if command == "run":
            parser.add_argument("--witness-output", type=Path)
    args = vars(cli.parse_args())
    commands[args.pop("command")](**args)
