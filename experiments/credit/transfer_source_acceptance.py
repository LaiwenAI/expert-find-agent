"""Evaluate frozen source-acceptance rules on DOI- and contributor-disjoint articles."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.credit import harvest_scoped_credit as publisher
from experiments.credit import heldout_source_windows as windows
from experiments.followir.core17_reranking import digest, write_json


def capture_transfer(folders, previous):
    old_dois, old_ids, old_names = publisher.prior(previous)
    rejected = Counter()
    eligible = {}
    for folder in folders:
        expected = json.loads((folder / "capture.json").read_bytes())["hashes"]
        found = {}
        for path in publisher.xml_sources(folder):
            raw = path.read_bytes()
            meta = publisher.meta(raw)
            doi = meta.findtext("article-id[@pub-id-type='doi']")
            found[doi] = sha256(raw).hexdigest()
            people = publisher.authors(meta)
            if doi in old_dois:
                rejected["doi"] += 1
            elif old_names.intersection(publisher.normalized_name(person) for person in people):
                rejected["name"] += 1
            elif old_ids.intersection(ident for ident, *_ in publisher.joined(people)):
                rejected["orcid"] += 1
            else:
                eligible.setdefault(doi, path)
        if found != expected:
            raise ValueError("Transfer publisher capture changed")
    with TemporaryDirectory() as temporary:
        dest = Path(temporary)
        for path in eligible.values():
            (dest / path.name).symlink_to(path.resolve())
        articles = windows.base.capture((dest,))
    return articles, rejected


def run(frozen, checkpoint, output, previous, transfer):
    import torch
    import math

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen_run = json.loads(frozen.read_bytes())
    articles, excluded = capture_transfer(transfer, previous)
    sources = {doi: article.digest for doi, article in articles.items()}
    generated = windows.base.cases(articles, len(articles))
    articles, rows = windows.prepare(articles, dict(sources=sources, queries=generated))
    index = windows.base.Index(articles)
    passages = {doi: windows.make_windows(article.abstract) for doi, article in articles.items()}
    weights = {term: math.log1p((len(articles) - len(items) + .5) / (len(items) + .5))
               for term, items in index.postings.items()}
    tokenizer, model, info = windows.base.reader_module.load_reader(checkpoint)
    if info["revision"] != frozen_run["checkpoint"]:
        raise ValueError("Source verifier checkpoint differs from calibration")
    read = windows.base.reader_module.reader_scorer(tokenizer, model, system=windows.SOURCE_SYSTEM)
    for arm in windows.ARMS:
        windows.observe(index, passages, weights, rows[0], "present", arm, read)

    states = ("present", "withheld")
    panels = {state: {arm: [] for arm in windows.ARMS} for state in states}
    for i, row in enumerate(rows):
        for j, state in enumerate(states):
            for arm in (windows.ARMS if (i + j) % 2 == 0 else windows.ARMS[::-1]):
                item = windows.observe(index, passages, weights, row, state, arm, read)
                panels[state][arm].append((row, item))

    result = {}
    for state, arm_observations in panels.items():
        result[state] = {}
        for arm, observations in arm_observations.items():
            rules = frozen_run["acceptance_gates"][arm]["0"]
            report = result[state][arm] = dict(
                queries=len(observations),
                recalled=sum(item["nominated"] for _, item in observations),
                proposed_correct=sum(item["proposed"] == row["doi"] for row, item in observations),
                reader_pairs=sum(item["calls"] for _, item in observations), policies={})
            for rule, r in rules.items():
                out = report["policies"][rule] = dict(cutoff=r["cutoff"], correct=0, wrong=0, supported_links=0,
                                                      wrong_source_links=0)
                for row, item in observations:
                    if windows.accepts(item, rule, r["cutoff"]):
                        is_correct = item["proposed"] == row["doi"]
                        requested = set(row["roles"])
                        links = sum(requested <= roles for roles in articles[item["proposed"]].roles)
                        out["correct" if is_correct else "wrong"] += 1
                        out["supported_links" if is_correct else "wrong_source_links"] += links
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        window_runner_sha256=digest(Path(windows.__file__)), frozen_sha256=digest(frozen), checkpoint=info["revision"],
        source_digests=sources, targets=len(rows), excluded=excluded, panels=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("frozen", "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--previous", nargs="+", type=Path, required=True)
    parser.add_argument("--transfer", nargs="+", type=Path, required=True)
    run(**vars(parser.parse_args()))
