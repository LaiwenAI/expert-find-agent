"""Collect exact-DOI MEDLINE MeSH coverage and replay same-person nomination."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from experiments.credit import role_score_gate as gate
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SNAPSHOTS = "https://www.nlm.nih.gov/mesh/meshhome.html"


def request(endpoint, **parameters):
    url = EUTILS + endpoint + ".fcgi?" + urlencode(dict(db="pubmed", **parameters))
    for attempt in range(3):
        try:
            with urlopen(url, timeout=22) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        sleep(2 ** attempt)
    raise RuntimeError("Unable to collect the PubMed batch")


def fetch(doi_batch):
    result = json.loads(request(
        "esearch", term=" OR ".join(f'"{doi}"[aid]' for doi in doi_batch),
        retmax=200, retmode="json",
    ))["esearchresult"]
    ids = set(result["idlist"])
    if len(ids) != int(result["count"]):
        raise ValueError("ESearch returned an incomplete batch")
    if not ids:
        return {}
    sleep(.4)
    root = ET.fromstring(request("efetch", id=",".join(sorted(ids)), retmode="xml"))
    rows = {}
    returned = set()
    for article in root.findall("PubmedArticle"):
        citation = article.find("MedlineCitation")
        pmid = citation.findtext("PMID")
        returned.add(pmid)
        matches = {node.text.casefold() for node in article.findall(
            "PubmedData/ArticleIdList/ArticleId[@IdType='doi']") if node.text} & set(doi_batch)
        for doi in matches:
            if doi in rows:
                raise ValueError(f"Ambiguous exact DOI: {doi}")
            rows[doi] = dict(
                pmid=pmid, status=citation.get("Status"),
                headings=sorted((
                    dict(ui=node.get("UI"), label=node.text, major=node.get("MajorTopicYN") == "Y")
                    for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                ), key=lambda heading: heading["ui"]),
            )
    if returned != ids:
        raise ValueError("EFetch did not return the matched PubMed identifiers")
    return rows


def capture(projection, output):
    if output.exists():
        raise ValueError("Provide a new snapshot path")
    dois = sorted({row["doi"].casefold() for row in json.loads(projection.read_bytes())["sources"]})
    records = {}
    for start in range(0, len(dois), 32):
        records.update(fetch(dois[start:start + 32]))
        print(f"Collected {start + 1} through {min(start + 32, len(dois))}", flush=True)
        sleep(.4)
    pilot.write_json(output, dict(
        projection_sha256=pilot.digest(projection), source=EUTILS, descriptors=SNAPSHOTS,
        records={doi: records.get(doi) for doi in dois},
    ))


def audit_keywords(snapshot, packet, review_summary, output):
    if output.exists() or pilot.digest(packet) != json.loads(review_summary.read_bytes())["packet_sha256"]:
        raise ValueError("Require the frozen review packet and a new output")
    records = json.loads(snapshot.read_bytes())["records"]
    terms = {h["label"].casefold() for row in records.values()
             if row is not None and row["status"] == "MEDLINE" for h in row["headings"]}
    published = json.loads(packet.read_bytes())["witnesses"]["keywords"]
    summary = {}
    eligible_topics = set()
    for mask, arms in published.items():
        summary[mask] = {}
        for arm, rows in arms.items():
            counts = Counter()
            for row in rows:
                topic = row["topic"].casefold()
                counts["requests"] += 1
                if topic not in terms:
                    continue
                eligible_topics.add(topic)
                counts["requests_with_exact_descriptor"] += 1
                for doi in row["citations"]:
                    record = records[doi.casefold()]
                    headings = record["headings"] if record and record["status"] == "MEDLINE" else []
                    counts["issued_on_exact_descriptor_requests"] += 1
                    counts["issued_with_medline_record"] += bool(headings)
                    counts["issued_matching_descriptor"] += any(h["label"].casefold() == topic for h in headings)
                    counts["issued_matching_major_descriptor"] += any(
                        h["major"] and h["label"].casefold() == topic for h in headings)
            summary[mask][arm] = counts
    pilot.write_json(output, dict(
        snapshot_sha256=pilot.digest(snapshot), packet_sha256=pilot.digest(packet),
        eligible_topic_count=len(eligible_topics), masks=summary,
    ))


def assign_topics(catalog, mesh, excluded):
    """Only major descriptors in explicitly MEDLINE-indexed records are positives."""
    catalog.topics.clear()
    label_by_ui = {}
    index_coverage = Counter()
    for index, article in enumerate(catalog.articles):
        record = mesh[article["doi"].casefold()]
        if record is None:
            index_coverage["missing"] += 1
            continue
        index_coverage[record["status"]] += 1
        if record["status"] != "MEDLINE":
            continue
        for term in record["headings"]:
            if term["major"] and term["ui"] not in excluded:
                if label_by_ui.setdefault(term["ui"], term["label"]) != term["label"]:
                    raise ValueError(f"Descriptor labels differ for {term['ui']}")
                catalog.topics[term["label"]].add(index)
    return index_coverage


def run(projection, snapshot, reference, gate_reference, embedding, reranker, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(snapshot.read_bytes())
    prior = json.loads(reference.read_bytes())["run"]
    gate_run = json.loads(gate_reference.read_bytes())["expanded_grid"]
    margin = gate_run["selected_margin"]
    if margin is None or gate_run["cutoff"] != holdout.CUTOFF:
        raise ValueError("Expected a locked nontrivial PLOS similarity floor")
    if pilot.digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Projection differs from the indexed catalog")
    articles = json.loads(projection.read_bytes())["sources"]
    if {row["doi"].casefold() for row in articles} != frozen["records"].keys():
        raise ValueError("Incomplete DOI inventory")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (prior["embedding"], prior["reranker"]):
        raise ValueError("Expected the frozen models")
    catalog = pilot.Catalog(articles, encoder, reader)
    complete = {i for i, a in enumerate(articles) if a["declaration_complete"]}
    catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
    # Exclude routine study-population descriptors even if marked major.
    excluded = {"D006801", "D051379", "D051381", "D005260", "D008297", "D000818"}
    coverage = assign_topics(catalog, frozen["records"], excluded)
    topics = sorted(t for t, ids in catalog.topics.items() if 4 <= len(ids) <= 40)
    if not topics:
        raise ValueError("No eligible MeSH topics in the pinned snapshot")
    catalog.verify(topics[0], catalog.rank(topics[0])[:8])
    eligible = set(topics)
    for term in tuple(catalog.topics):
        if term not in eligible:
            del catalog.topics[term]
    controls = tuple((arm, None) for arm in gate.CONTROL)
    measurements = gate.collect(catalog, 4, (*controls, ("gated_four", margin)), cache_scores=False)
    results = {
        pct: {arm: dict(
            injected=sum(row["injected"] for row in rows),
            requests_answered=sum(bool(row["issued"]) for row in rows),
            compared={base: holdout.analyze(rows, arms[base]) for base in ("topic_four", "topic_eight")},
        ) for arm, rows in arms.items()}
        for pct, arms in measurements.items()
    }
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), snapshot_sha256=pilot.digest(snapshot),
        projection_sha256=pilot.digest(projection), reference_sha256=pilot.digest(reference),
        coverage=coverage, declaration_indexed={pct: len(ids) for pct, ids in catalog.visible.items()},
        locked_margin=margin, gate_reference_sha256=pilot.digest(gate_reference),
        eligible_topics=len(topics), results=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    commands = {
        "capture": (capture, ("projection", "output")),
        "audit": (audit_keywords, ("snapshot", "packet", "review_summary", "output")),
        "run": (run, ("projection", "snapshot", "reference", "gate_reference", "embedding", "reranker", "output")),
    }
    for name, (_, arguments) in commands.items():
        parser = sub.add_parser(name)
        for argument in arguments:
            parser.add_argument(argument, type=Path)
    args = vars(cli.parse_args())
    commands[args.pop("command")][0](**args)
