"""Evaluate author attribution in original publisher contribution notes."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from hashlib import sha256
import os
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET


TOPICS = {
    "calibration": ("graph clustering", "image segmentation", "network topology",
                    "data mining", "community detection", "wireless sensor"),
    "test": ("wind power", "wind energy", "photovoltaic", "electricity demand", "renewable energy"),
}
EXTRA_TEST_TOPICS = ("solar cell", "solar energy", "wind turbine", "energy consumption",
                     "energy efficiency", "power generation", "solar irradiance",
                     "energy production", "electric power")
NUMBER = 42
LABELS = {
    "conceived and designed the experiments": "design",
    "performed the experiments": "execution",
    "analyzed the data": "analysis",
    "wrote the paper": "writing",
    "contributed reagents/materials/analysis tools": "resources",
}
PHRASES = {
    "design": ("planned how the experiments would be set up",
               "helped formulate the study and design its experiments"),
    "execution": ("carried out the experiments",
                  "personally helped conduct the experimental work"),
    "analysis": ("processed or interpreted the resulting data",
                 "contributed to analysis of the collected observations"),
    "writing": ("participated in writing the article",
                "helped draft the manuscript reporting the work"),
    "resources": ("provided materials or analysis tools used in the study",
                  "made study resources, materials or analytic tools available"),
}
INSTRUCTION = ("Does the contribution note explicitly credit the named author with every requested task? "
               "A task credited only to another author does not count. Partial support is insufficient.")
ARMS = ("broad", "bound", "scope_first", "reader_4b")
INITIAL = re.compile("[A-Z]{2,8}")
NAME_PARTS = re.compile(r"[^\W\d_]+")
SENTENCE = re.compile(r"(?<=\.)\s+")
DOI = re.compile(r"10\.1371/journal\.pone\.\d+")


def fetch(url):
    with urlopen(url, timeout=18) as response:
        return response.read()


def stable(*items):
    return sha256(json.dumps(items, sort_keys=True).encode()).digest()


def search(topics, start):
    query = ('journal:"PLOS ONE" AND publication_date:[2010-01-01T00:00:00Z TO 2015-12-31T23:59:59Z] '
             'AND title:(' + " OR ".join(f'"{word}"' for word in topics) + ')')
    return "https://api.plos.org/search?" + urlencode(
        dict(q=query, rows=300, start=start, fl="id,title", sort="publication_date asc,id asc", wt="json")
    )


def inventory(topics):
    rows = {}
    receipts = {}
    start = 0
    while True:
        url = search(topics, start)
        raw = fetch(url)
        receipts[url] = sha256(raw).hexdigest()
        result = json.loads(raw)["response"]
        documents = result["docs"]
        for row in documents:
            title = row.get("title", "").casefold()
            if DOI.fullmatch(row["id"]) and any(term in title for term in topics):
                rows[row["id"]] = row["title"]
        start += len(documents)
        if not documents or start >= result["numFound"]:
            return rows, receipts


def paper_uri(doi):
    return "https://journals.plos.org/plosone/article/file?" + urlencode(dict(id=doi, type="manuscript"))


def note(raw, doi):
    article = ET.fromstring(raw).find("./front/article-meta")
    if article is None or article.findtext("article-id[@pub-id-type='doi']") != doi:
        raise ValueError("Unexpected publisher DOI")
    license_node = article.find("./permissions/license")
    url = "" if license_node is None else license_node.get("{http://www.w3.org/1999/xlink}href", "")
    license_text = "" if license_node is None else " ".join(license_node.itertext()).casefold()
    explicit_open = (re.fullmatch(r"https?://creativecommons\.org/(?:licenses/by|publicdomain/zero)/[0-9.]+/?", url)
                     or "creative commons attribution license" in license_text)
    if not explicit_open:
        return None
    matches = article.findall("./author-notes/fn[@fn-type='con']")
    if len(matches) != 1:
        return None
    text = " ".join(" ".join(matches[0].itertext()).split())
    byline = [NAME_PARTS.findall(
        f"{name.findtext('given-names') or ''} {name.findtext('surname') or ''}")
        for name in article.findall("./contrib-group/contrib[@contrib-type='author']/name")]
    names_by_initials = Counter("".join(part[0] for part in name).upper() for name in byline)
    names = {" ".join(part.casefold() for part in name) for name in byline if len(name) >= 2}
    unambiguous = {name for name, count in names_by_initials.items() if count == 1 and INITIAL.fullmatch(name)}
    labels = {}
    for sentence in SENTENCE.split(text):
        role, sep, block = sentence.partition(":")
        label = LABELS.get(role.strip().casefold())
        if not sep or label is None:
            continue
        tokens = block.strip().rstrip(".").split()
        if label in labels:
            return None
        if tokens and all(INITIAL.fullmatch(name) for name in tokens):
            labels[label] = set(tokens) & unambiguous
    return url, text, labels, names


def choose_case(doi, labels):
    variants = []
    for (lhs, a), (rhs, b) in combinations(sorted(labels.items()), 2):
        both, partial = a & b, a ^ b
        variants.extend(((lhs, rhs), good, other) for good in sorted(both) for other in sorted(partial))
    return min(variants, key=lambda item: stable(doi, *item)) if variants else None


def download(doi):
    for _ in range(2):
        try:
            return doi, fetch(paper_uri(doi)), None
        except (OSError, TimeoutError) as error:
            problem = type(error).__name__
    return doi, None, problem


def take(folder, candidates):
    checks, failures = {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for doi, raw, failure in pool.map(download, candidates):
            if raw is None:
                failures[doi] = failure
                continue
            checks[doi] = sha256(raw).hexdigest()
            (folder / (doi.rsplit(".", 1)[-1] + ".xml")).write_bytes(raw)
    return checks, failures


def capture(output):
    if output.exists():
        raise FileExistsError(output)
    candidates = {}
    for split, topics in TOPICS.items():
        candidates[split] = inventory(topics)
    overlap = candidates["calibration"][0].keys() & candidates["test"][0].keys()
    output.mkdir()
    manifest = {}
    for split, (articles, receipts) in candidates.items():
        chosen = sorted(articles.keys() - overlap, key=lambda doi: stable(split, doi))[:NUMBER]
        folder = output / split
        folder.mkdir()
        checks, failures = take(folder, chosen)
        manifest[split] = dict(queries=receipts, candidates=len(articles), selected=chosen, checksums=checks,
                               failed=failures, overlap_removed=len(overlap))
        print(split, len(articles), len(checks), str(failures), flush=True)
    (output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")


def extend(snapshot):
    receipt = snapshot / "capture.json"
    manifest = json.loads(receipt.read_text())
    test = manifest["test"]
    if "extension_candidates" in test:
        raise ValueError("The test capture already includes its extension")
    articles, queries = inventory(EXTRA_TEST_TOPICS)
    existing = set(test["selected"]) | set(manifest["calibration"]["selected"])
    eligible = articles.keys() - existing
    chosen = sorted(eligible, key=lambda doi: stable("test-extension", doi))[:NUMBER - len(test["selected"])]
    received, failed = take(snapshot / "test", chosen)
    test["selected"].extend(chosen)
    test["checksums"].update(received)
    test["failed"].update(failed)
    test["queries"].update(queries)
    test["extension_candidates"] = len(eligible)
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    print("test extension", len(eligible), len(received), str(failed), flush=True)


def restore(receipt, snapshot):
    if snapshot.exists():
        raise FileExistsError(snapshot)
    content = receipt.read_bytes()
    manifest = json.loads(content)
    snapshot.mkdir()
    for split in ("calibration", "test"):
        expected = manifest[split]["checksums"]
        folder = snapshot / split
        folder.mkdir()
        received, failed = take(folder, sorted(expected))
        if failed or received != expected:
            raise ValueError(f"Cannot restore the pinned publisher files for {split}")
    (snapshot / "capture.json").write_bytes(content)


def prepare(snapshot):
    manifest = json.loads((snapshot / "capture.json").read_text())
    panels = {}
    used_names = set()
    for split in ("calibration", "test"):
        inventory = manifest[split]
        rows, omitted = [], Counter()
        for doi, checksum in inventory["checksums"].items():
            raw = (snapshot / split / (doi.rsplit(".", 1)[-1] + ".xml")).read_bytes()
            if sha256(raw).hexdigest() != checksum:
                raise ValueError("Changed publisher note capture")
            parsed = note(raw, doi)
            if parsed is None:
                omitted["missing_supported_format_or_license"] += 1
                continue
            _, text, labels, names = parsed
            if split == "calibration":
                used_names.update(names)
            elif not used_names.isdisjoint(names):
                omitted["shared_byline_name"] += 1
                continue
            case = choose_case(doi, labels)
            if case is None:
                omitted["no_two_duty_contrast"] += 1
                continue
            tasks, good, other = case
            rows.append((doi, text, tasks, good, other))
        panels[split] = rows, dict(omitted)
    return panels


def scope(text, initials):
    matcher = re.compile(rf"(?<![\w-]){re.escape(initials)}(?![\w-])")
    return " ".join(line for line in SENTENCE.split(text) if matcher.search(line))


def measure(rows, variant, reranker, judge, reader, tokenizer, labels):
    from collections import defaultdict
    out = defaultdict(list)
    for doi, text, tasks, good, other in rows:
        ordered = (other, good) if stable(doi)[0] % 2 else (good, other)
        query = {name: f"Target author initials: {name}\nRequired contributions: "
                 + "; ".join(PHRASES[role][variant] for role in tasks) for name in ordered}
        for arm in ARMS:
            start = perf_counter()
            documents = [scope(text, name) if arm == "scope_first" else text for name in ordered]
            if arm == "reader_4b":
                # The shared open reader takes one combined query and a list of documents.
                # Put the two distinct queries in their respective document payloads.
                values = judge(reader, tokenizer, labels, "Apply the decision separately to this author.",
                               [f"{query[name]}\n\n{doc}" for name, doc in zip(ordered, documents, strict=True)])
            else:
                values, clipped = reranker.score_pairs(
                    [(query[name], doc) for name, doc in zip(ordered, documents, strict=True)],
                    task="Find a relevant contributor to the requested tasks." if arm == "broad" else INSTRUCTION)
                if clipped:
                    raise ValueError("Legacy contribution notes should fit the short context")
            scores = dict(zip(ordered, values, strict=True))
            out[arm].append((scores[good], scores[other], (perf_counter() - start) * 1000))
    return out


def run(snapshot, reranker_path, reader_path, output):
    import numpy as np
    import torch
    from expert_search.statistics import paired_interval
    from experiments.credit.claim_bound_attribution import judge, operating_point
    from experiments.followir.core17_reranking import digest, write_json
    from experiments.followir.core17_verifier import Reranker
    from experiments.qasper.paired_reader import load_reader

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh results in a Slurm CUDA allocation")
    panels = prepare(snapshot)
    cal, test = panels["calibration"][0], panels["test"][0]
    if min(len(cal), len(test)) < 12 or {r[0] for r in cal} & {r[0] for r in test}:
        raise ValueError("Insufficient disjoint publication groups")
    reranker = Reranker(reranker_path)
    tokenizer, reader, reader_info = load_reader(reader_path)
    code = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(item) != 1 for item in code) or code[0] == code[1]:
        raise ValueError("Require two distinct single-token decisions")
    engines = (reranker, judge, reader, tokenizer, [item[0] for item in code])
    measure(cal[:1], 0, *engines)
    calibration, evaluation = measure(cal, 0, *engines), measure(test, 1, *engines)
    methods = {}
    for arm in ARMS:
        cases = evaluation[arm]
        wrong = sorted((b for _, b, _ in calibration[arm]), reverse=True)
        threshold = wrong[int(.10 * len(wrong))]
        methods[arm] = dict(
            correct_order=sum(a > b for a, b, _ in cases), ties=sum(a == b for a, b, _ in cases),
            threshold=threshold, calibration_false=sum(b > threshold for b in wrong),
            paired_delta=paired_interval([int(a > b) - int(x > y)
                                           for (a, b, _), (x, y, _) in zip(cases, evaluation["broad"], strict=True)]),
            p95_ms=float(np.quantile([ms for _, _, ms in cases], .95)),
            frozen=operating_point([(a,b) for a,b,_ in cases], threshold),
        )
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            script_sha256=digest(Path(__file__)), capture_sha256=digest(snapshot / "capture.json"),
                            reranker_revision=reranker.provenance["revision"], reader_revision=reader_info["revision"],
                            panels={key: dict(count=len(x), excluded=miss,
                                              included_dois=[row[0] for row in x]) for key, (x,miss) in panels.items()},
                            report=methods))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("output", type=Path)
    extension = commands.add_parser("extend")
    extension.add_argument("snapshot", type=Path)
    reproduce = commands.add_parser("restore")
    reproduce.add_argument("receipt", type=Path)
    reproduce.add_argument("snapshot", type=Path)
    score = commands.add_parser("run")
    for name in ("snapshot", "reranker_path", "reader_path", "output"):
        score.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    {"capture": capture, "extend": extend, "restore": restore, "run": run}[args.pop("mode")](**args)
