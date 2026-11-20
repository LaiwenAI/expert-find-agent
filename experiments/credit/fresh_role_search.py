"""Replay locked role heads and structured author binding in a fresh corpus."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.error import URLError
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import harvest_scoped_credit as plos
from experiments.credit.plos_role_inventory import declared_roles, ROLES
from experiments.followir.core17_reranking import digest, write_json

ARMS = (
    ("single_article", "single_clause_cross_entropy", False),
    ("single_person", "single_clause_cross_entropy", True),
    ("near_article", "near_role_swap_cross_entropy", False),
    ("near_person", "near_role_swap_cross_entropy", True),
    ("reader_person", "reader", True),
)
PREFIX = plos.DOI_PREFIX
DEPTH = 4
TOP_K = 5
TOPICS = {
    "education": 'title:("higher education" OR "university teaching")',
    "transport": 'title:("public transportation" OR "public transit" OR "urban mobility" OR "ride-hailing")',
    "housing": 'title:("housing prices" OR "housing affordability" OR "housing markets" OR "rental housing")',
}
ROLE_IDS = {role: index for index, role in enumerate(ROLES)}


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    dois = set()
    for name, topic in TOPICS.items():
        date_range = "2025-01-01T00:00:00Z TO 2026-10-31T23:59:59Z"
        query = f'journal:"PLOS ONE" AND {topic} AND publication_date:[{date_range}]'
        url = "https://api.plos.org/search?" + urlencode(dict(
            q=query, fq=["doc_type:full", 'article_type:"Research Article"'],
            fl="id,title_display,publication_date", rows=25, wt="json",
            sort="publication_date desc,id asc",
        ), doseq=True)
        path = folder / f"{name}-search.json"
        if not path.exists():
            path.write_bytes(plos.read_url(url))
        docs = json.loads(path.read_bytes())["response"]["docs"]
        dois.update(row["id"] for row in docs if plos.is_article(row["id"]))

    def fetch(doi):
        try:
            plos.get_xml(folder, doi)
        except (ValueError, ET.ParseError, URLError, TimeoutError) as error:
            return doi, str(error)
        return doi, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        failures = {doi: error for doi, error in pool.map(fetch, sorted(dois)) if error}
    write_json(folder / "fetch-errors.json", failures)


def capture(folder, wording_path, output, previous):
    prior_dois, prior_ids, prior_names = plos.prior(previous)
    if not prior_dois or not prior_names:
        raise ValueError("Missing prior articles")
    inventory = sorted(folder.glob("*-search.json"))
    records = {}
    for path in inventory:
        items = json.loads(path.read_bytes())["response"]["docs"]
        records.update({item["id"]: item["title_display"] for item in items})
    sources, rejected = {}, {}
    admitted_names, admitted_ids = set(), set()
    for doi, title in sorted(records.items()):
        if not plos.is_article(doi) or title.casefold().startswith(("retracted:", "retraction:", "correction:")):
            rejected[doi] = "not an eligible research article"
            continue
        path = xml_path(folder, doi)
        if not path.exists():
            rejected[doi] = "missing capture"
            continue
        raw = path.read_bytes()
        metadata = plos.meta(raw)
        authors = plos.authors(metadata)
        names = {plos.normalized_name(author) for author in authors}
        ids = {identity for identity, _, _ in plos.joined(authors)}
        roles = declared_roles(raw, doi)
        if None in names or len(authors) != len(names) or len(roles) < 2 or not all(roles):
            rejected[doi] = "ambiguous author or incomplete roles"
            continue
        if doi in prior_dois or names & prior_names or ids & prior_ids:
            rejected[doi] = "overlap with prior sources"
            continue
        if names & admitted_names or ids & admitted_ids:
            rejected[doi] = "byline overlap with earlier admitted article"
            continue
        if metadata.find("./abstract") is None or metadata.find("./title-group/article-title") is None:
            rejected[doi] = "missing topic text"
            continue
        admitted_names.update(names)
        admitted_ids.update(ids)
        sources[doi] = sha256(raw).hexdigest()
    wording = json.loads(wording_path.read_bytes())
    queries = sorted(PREFIX + suffix for suffix in wording["topics"] if PREFIX + suffix in sources)
    if not queries:
        raise ValueError("No eligible topic queries")
    write_json(output, dict(
        sources=sources,
        queries=queries, rejected=rejected, prior_dois=len(prior_dois),
        prior_orcids=len(prior_ids), prior_byline_names=len(prior_names),
        prior_folders=sorted("/".join(path.parts[-2:]) if path.name in ("calibration", "test") else path.name
                             for path in previous),
        prior_doi_sha256=sha256("\n".join(sorted(prior_dois)).encode()).hexdigest(),
        search={path.name: digest(path) for path in inventory},
        wording_sha256=digest(wording_path),
    ))


def xml_path(folder, doi):
    if not plos.is_article(doi):
        raise ValueError("Expected a PLOS ONE DOI")
    return folder / f"pone.{doi.removeprefix(PREFIX)}.xml"


def article(folder, doi, checksum):
    path = xml_path(folder, doi)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError(f"Changed author evidence: {doi}")
    root = ET.fromstring(raw)
    metadata = root.find("./front/article-meta")
    title = metadata.find("./title-group/article-title")
    abstract = metadata.find("./abstract")
    if title is None or abstract is None:
        raise ValueError(f"Missing topic text: {doi}")
    text = "\n".join(" ".join(" ".join(node.itertext()).split()) for node in (title, abstract))
    return text, declared_roles(raw, doi)


def nominate(shortlist, declarations, scores, threshold, bind_person):
    output = []
    for index in shortlist:
        article_roles = declarations[index]
        options = sorted(
            ((-scores[ROLE_IDS[role]], owner, role)
             for owner, roles in enumerate(article_roles) for role in roles
             if scores[ROLE_IDS[role]] > threshold),
        )
        if bind_person:
            cited = {}
            for _, owner, role in options:
                cited.setdefault(owner, (owner, role))
        elif options:
            _, owner, role = options[0]
            cited = dict.fromkeys(range(len(article_roles)), (owner, role))
        else:
            cited = {}
        output.extend((index, candidate, owner, role) for candidate, (owner, role) in cited.items())
        if len(output) >= TOP_K:
            break
    return output[:TOP_K]


def metrics(output, target, wanted):
    return dict(
        supported=sum(index == target and owner == person and role == wanted
                      for index, person, owner, role in output),
        wrong_owner=sum(owner != person for _, person, owner, _ in output),
        wrong_role=sum(role != wanted for _, _, _, role in output),
        correct_role_wrong_owner=sum(role == wanted and owner != person
                                     for _, person, owner, role in output),
        wrong_role_correct_owner=sum(role != wanted and owner == person
                                     for _, person, owner, role in output),
        other_source=sum(index != target for index, *_ in output),
        emitted=len(output),
    )


def run(folder, fixture_path, wording_path, training, reference, embedding, reranker, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.factored_article_role_verification import ROLE_TASK
    from experiments.credit.factorial_role_training import fit
    from experiments.credit.train_role_decision import encode
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require Slurm CUDA and a new output")
    fixture = json.loads(fixture_path.read_bytes())
    previous = json.loads(reference.read_bytes())
    if digest(wording_path) != fixture["wording_sha256"] or digest(training) != previous["training_sha256"]:
        raise ValueError("Changed frozen inputs")
    wording = json.loads(wording_path.read_bytes())
    dois = sorted(fixture["sources"])
    texts, declarations = zip(*(article(folder, doi, fixture["sources"][doi]) for doi in dois), strict=True)
    encoder = QwenTorchEmbedding(embedding)
    reader = Reranker(reranker)
    models = dict(embedding=verify_checkpoint(embedding)["revision"], reader=reader.provenance["revision"])
    if models != previous["checkpoints"]:
        raise ValueError("Changed model revisions")
    weights, _ = fit(encoder, training)
    vectors = encoder.encode(list(texts))
    queries = []
    for doi in fixture["queries"]:
        expected = dois.index(doi)
        topic = wording["topics"][doi.removeprefix(PREFIX)]
        counts = Counter(role for roles in declarations[expected] for role in roles)
        for role, phrase in wording["activities"].items():
            if 0 < counts[role] < len(declarations[expected]):
                queries.append((expected, topic, role, phrase))
    if not queries:
        raise ValueError("Missing contrasting query set")
    rows = {name: [] for name, _, _ in ARMS}
    timings = {name: [] for name in rows}
    hit = []
    for i, (target, topic, wanted, phrase) in enumerate([queries[0], *queries]):
        order = i % len(ARMS)
        for name, model, link in ARMS[order:] + ARMS[:order]:
            started = perf_counter()
            topic_vector = encoder.encode([topic], query=True)[0]
            shortlist = np.argsort(-(vectors @ topic_vector), kind="stable")[:DEPTH].tolist()
            if model == "reader":
                scores, clipped = reader.score_pairs([(phrase, role) for role in ROLES], task=ROLE_TASK)
                if clipped or not np.isfinite(scores).all():
                    raise ValueError("Invalid reader output")
            else:
                with torch.inference_mode():
                    scores = (weights[model] @ encode(encoder, [phrase])[0]).cpu().tolist()
            response = nominate(shortlist, declarations, scores, previous["thresholds"][model], link)
            elapsed = (perf_counter() - started) * 1000
            if i:
                rows[name].append(metrics(response, target, wanted))
                timings[name].append(elapsed)
                if name == "single_person":
                    hit.append(int(target in shortlist))
    groups = [query[0] for query in queries]
    summaries = {name: dict(
        counts={key: sum(row[key] for row in outcomes) for key in outcomes[0]},
        p95_ms=float(np.quantile(timings[name], .95)),
        issued=sum(row["emitted"] > 0 for row in outcomes),
        any_supported=sum(row["supported"] > 0 for row in outcomes),
        clean_focus=sum(row["supported"] == row["emitted"] > 0 for row in outcomes),
    ) for name, outcomes in rows.items()}
    differences = {}
    for a, b in (("near_person", "single_person"), ("near_person", "near_article"),
                 ("single_person", "single_article"), ("near_person", "reader_person")):
        differences[f"{a}_vs_{b}"] = {
            key: clustered_paired_interval(
                [(x[key] - y[key]) / TOP_K for x, y in zip(rows[a], rows[b], strict=True)], groups,
            )
            for key in ("supported", "wrong_owner", "wrong_role", "other_source")
        }
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(queries), papers=len(dois), target_papers=len(set(groups)),
        queries_by_role=Counter(query[2] for query in queries),
        target_in_shortlist=sum(hit), shortlist_depth=DEPTH, top_k=TOP_K,
        fixture_sha256=digest(fixture_path), reference_sha256=digest(reference),
        code_sha256=digest(Path(__file__)), arms=summaries, paired_differences=differences,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    modes.add_parser("download").add_argument("folder", type=Path)
    builder = modes.add_parser("capture")
    for name in ("folder", "wording_path", "output"):
        builder.add_argument(name, type=Path)
    builder.add_argument("previous", type=Path, nargs="+")
    runner = modes.add_parser("run")
    for name in ("folder", "fixture_path", "wording_path", "training", "reference", "embedding", "reranker", "output"):
        runner.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    fn = {"download": download, "capture": capture, "run": run}[args.pop("command")]
    fn(**args)
